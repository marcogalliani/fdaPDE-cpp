// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

/* What does a coarser time grid actually cost?
=============================================================================================

Coarsening the fit grid removes data and coarsens the control at the same time, so a raw error-versus-m
curve conflates two effects. This benchmark separates them with an extra column: `integ_err` is the fit to
NOISELESS data at a very large lambda, which drives the control to zero and leaves the scheme's integration
error alone. Comparing the two columns says which of the two is binding at each resolution.

The conclusion it records: on these problems the coarse-grid penalty is overwhelmingly INTEGRATION order,
not control resolution. GL1 is integration-limited on coarse grids (total error ~ integ_err); GL2 and GL3
are noise-limited everywhere, and differ from each other by almost nothing. Choosing a higher-order Gauss
scheme is what makes a coarse grid affordable -- not enriching the control.

Accuracy is measured through the solver's continuous eval() on a grid-independent set of query points,
which is the only way coarse and fine fits are comparable at all. */

namespace {
using namespace fdapde::bench;

// best achievable accuracy over a lambda grid ("oracle lambda"), averaged over paired noise replicates.
// Selecting lambda per configuration isolates representational capability from lambda selection, which is
// a separate problem; these numbers are therefore upper bounds on what a real GCV-driven fit would give.
struct sweep_result {
    double lambda = 0, err = 1e300, iters = 0, ms = 0;
    int converged = 0, n = 0;
};

sweep_result oracle_sweep(int degree, const vector_t& time, const matrix_t& Ytrue, const eval_grid& g,
  const std::vector<double>& lambdas, double sigma, int reps) {
    sweep_result best;
    for (double lambda : lambdas) {
        double acc = 0, it = 0, ms = 0;
        int ok = 0, conv = 0;
        for (int r = 0; r < reps; ++r) {
            Triangulation<1, 1> T(time);
            BsSpace Vh(T, degree, std::vector<int>(T.n_nodes(), degree));   // C0 at every node
            fdapde::bs_ls_ode penalty(nonlinear_field {}, Vh);
            fdapde::internals::bs_ls_ode solver;
            solver.discretize(penalty.get());
            solver.analyze_data(time, add_noise(Ytrue, sigma, r));
            const auto t0 = std::chrono::steady_clock::now();
            solver.fit(lambda);
            const auto t1 = std::chrono::steady_clock::now();
            if (!solver.trajectory().allFinite()) { continue; }
            const matrix_t E = solver.eval(g.q);
            if (!E.allFinite()) { continue; }
            acc += rms(E, g.truth);
            it += solver.n_iter();
            ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            conv += solver.converged() ? 1 : 0;
            ++ok;
        }
        if (ok == 0) { continue; }
        if (acc / ok < best.err) { best = {lambda, acc / ok, it / ok, ms / ok, conv, ok}; }
    }
    return best;
}

// integration error alone: noiseless data, penalty large enough that the control is driven to zero
double integration_error(int degree, const vector_t& time, const matrix_t& Ytrue, const eval_grid& g) {
    Triangulation<1, 1> T(time);
    BsSpace Vh(T, degree, std::vector<int>(T.n_nodes(), degree));   // C0 at every node
    fdapde::bs_ls_ode penalty(nonlinear_field {}, Vh);
    fdapde::internals::bs_ls_ode solver;
    solver.discretize(penalty.get());
    solver.analyze_data(time, Ytrue);
    solver.fit(1e8);
    if (!solver.trajectory().allFinite()) { return std::nan(""); }
    const matrix_t E = solver.eval(g.q);
    return E.allFinite() ? rms(E, g.truth) : std::nan("");
}

}   // namespace

FDAPDE_BENCHMARK(grid_refinement, "accuracy and cost versus time-grid resolution, per Gauss scheme") {
    const double T = 2.0, sigma = 0.02, amp = 0.0;   // correctly specified: isolates discretisation
    vector_t y0(2);
    y0 << 0.5, -0.3;
    const eval_grid g(nonlinear_field {}, y0, T, 401, 10);
    const std::vector<int> ms = opt.full ? std::vector<int> {4, 6, 11, 21, 41, 81} : std::vector<int> {6, 11, 21, 41};
    const std::vector<double> lambdas = {1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e6};

    table t({"scheme", "s", "m", "dt", "best lambda", "rmse", "integ_err", "integ/rmse", "iters", "ms/fit"});
    std::vector<std::vector<double>> rmse_by_scheme, integ_by_scheme;

    // one study per trajectory degree; the degree is also the stage count of the scheme it selects
    auto study = [&](const char* name, int degree) {
        const int s = degree;
        std::vector<double> rmses, integs;
        for (std::size_t gi = 0; gi < ms.size(); ++gi) {
            const int m = ms[gi];
            const vector_t time = vector_t::LinSpaced(m, 0.0, T);
            const matrix_t Ytrue = g.at(time, T);
            const sweep_result b = oracle_sweep(degree, time, Ytrue, g, lambdas, sigma, opt.reps);
            const double ie = integration_error(degree, time, Ytrue, g);
            t.row({gi == 0 ? name : "", gi == 0 ? num(s) : "", num(m), fix(T / (m - 1), 4), sci(b.lambda, 0),
                   sci(b.err), sci(ie), fix(ie / b.err, 3), fix(b.iters, 1), fix(b.ms, 2)});
            rmses.push_back(b.err);
            integs.push_back(ie);
        }
        t.rule();
        rmse_by_scheme.push_back(rmses);
        integ_by_scheme.push_back(integs);
    };
    study("degree 1 (GL1)", 1);
    study("degree 2 (GL2)", 2);
    study("degree 3 (GL3)", 3);
    if (!opt.quiet) {
        std::cout << "  correctly specified, sigma = " << sigma << ", " << opt.reps
                  << " paired replicates, oracle lambda.\n"
                  << "  integ_err: noiseless fit at lambda = 1e8 (control driven to zero) -> integration error alone.\n"
                  << "  measured through eval() on a fixed 401-point query grid, so rows are comparable across m.\n";
        t.print();
    }
    (void)amp;

    // GL1 is integration-limited on the coarsest grid: its integration error is a large share of its total
    rep.checkf("GL1 is integration-limited on the coarsest grid", integ_by_scheme[0][0] / rmse_by_scheme[0][0] > 0.5,
               "m=%d: integ_err %.3e of rmse %.3e (%.0f%%)", ms[0], integ_by_scheme[0][0], rmse_by_scheme[0][0],
               100 * integ_by_scheme[0][0] / rmse_by_scheme[0][0]);
    // GL2 is not: its integration error is a small share, so its remaining error is noise, not discretisation
    rep.checkf("GL2 is not integration-limited even on the coarsest grid",
               integ_by_scheme[1][0] / rmse_by_scheme[1][0] < 0.5, "m=%d: integ_err %.3e of rmse %.3e (%.0f%%)", ms[0],
               integ_by_scheme[1][0], rmse_by_scheme[1][0], 100 * integ_by_scheme[1][0] / rmse_by_scheme[1][0]);
    // a higher-order scheme is what buys a coarse grid: GL2 on the coarsest grid beats GL1 there outright
    rep.checkf("GL2 beats GL1 on the coarsest grid", rmse_by_scheme[1][0] < rmse_by_scheme[0][0],
               "m=%d: GL1 %.4e, GL2 %.4e (%.1fx)", ms[0], rmse_by_scheme[0][0], rmse_by_scheme[1][0],
               rmse_by_scheme[0][0] / rmse_by_scheme[1][0]);
    // but GL3 buys almost nothing over GL2: once integration is not binding, order stops mattering
    const double gl2_gl3 = std::abs(rmse_by_scheme[1].back() - rmse_by_scheme[2].back()) / rmse_by_scheme[1].back();
    rep.checkf("GL3 adds nothing over GL2 once integration is not binding", gl2_gl3 < 0.05,
               "m=%d: GL2 %.4e, GL3 %.4e (%.2f%% apart)", ms.back(), rmse_by_scheme[1].back(),
               rmse_by_scheme[2].back(), 100 * gl2_gl3);
}
