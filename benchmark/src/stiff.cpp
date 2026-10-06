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

/* Stiff dynamics: when does stiffness actually cost anything?
=============================================================================================

Two regimes of the same stiff system, distinguished only by the initial condition:

  ON the slow manifold  (y0(0) = sin 0 = 0). The equation is stiff but the solution is smooth. A-stability
  is doing its job -- the fast mode is damped rather than resolved, the slow manifold is tracked, and every
  Gauss scheme is noise-limited even at k*dt = 40. Stiffness is a non-event.

  OFF the slow manifold (y0(0) = 1). A fast layer of width ~1/k appears that no grid here resolves. Gauss
  schemes are A-stable but NOT L-stable -- |R(z)| -> 1 as z -> -infinity -- so the layer error is propagated
  rather than damped. Errors jump one to two orders of magnitude, convergence order collapses, and higher
  order genuinely pays for the first time in this benchmark suite.

The second regime is also where the reduced solve struggles: the best lambda becomes small, and small
lambda is where the control preconditioner is worst conditioned, because its scale assumes B_t ~ dt*I while
a stiff component is damped by 1/(1 + O(k*dt)). The reported iteration counts are the visible symptom.

Practical reading: if unresolved stiff transients are expected, the missing ingredient is an L-stable
scheme (Radau IIA), which no Gauss method can provide at any order. */

namespace {
using namespace fdapde::bench;

struct stiff_result {
    double lambda = 0, err = 1e300, integ = 0, iters = 0;
    int converged = 0;
};

stiff_result stiff_sweep(int degree, const stiff_field& f, const vector_t& time, const matrix_t& Ytrue,
  const eval_grid& g, const std::vector<double>& lambdas, double sigma, int reps) {
    stiff_result best;
    // integration error alone (noiseless, control driven to zero)
    {
        Triangulation<1, 1> T(time);
        BsSpace Vh(T, degree, std::vector<int>(T.n_nodes(), degree));   // C0 at every node
        fdapde::bs_ls_ode penalty(f, Vh);
        fdapde::internals::bs_ls_ode solver;
        solver.discretize(penalty.get());
        solver.analyze_data(time, Ytrue);
        solver.fit(1e8);
        if (solver.trajectory().allFinite()) {
            const matrix_t E = solver.eval(g.q);
            if (E.allFinite()) { best.integ = rms(E, g.truth); }
        }
    }
    for (double lambda : lambdas) {
        double acc = 0, it = 0;
        int ok = 0, conv = 0;
        for (int r = 0; r < reps; ++r) {
            Triangulation<1, 1> T(time);
            BsSpace Vh(T, degree, std::vector<int>(T.n_nodes(), degree));   // C0 at every node
            fdapde::bs_ls_ode penalty(f, Vh);
            fdapde::internals::bs_ls_ode solver;
            solver.discretize(penalty.get());
            solver.analyze_data(time, add_noise(Ytrue, sigma, r));
            solver.fit(lambda);
            if (!solver.trajectory().allFinite()) { continue; }
            const matrix_t E = solver.eval(g.q);
            if (!E.allFinite()) { continue; }
            acc += rms(E, g.truth);
            it += solver.n_iter();
            conv += solver.converged() ? 1 : 0;
            ++ok;
        }
        if (ok == 0) { continue; }
        if (acc / ok < best.err) {
            const double integ = best.integ;
            best = {lambda, acc / ok, integ, it / ok, conv};
        }
    }
    return best;
}

}   // namespace

FDAPDE_BENCHMARK(stiff, "stiff dynamics on and off the slow manifold (k = 200)") {
    const double T = 2.0, sigma = 0.02, k = 200.0;
    const stiff_field f {k};
    const std::vector<int> ms = opt.full ? std::vector<int> {11, 21, 41, 81, 161} : std::vector<int> {11, 21, 41, 81};
    const std::vector<double> lambdas = {1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e6};

    // both regimes differ ONLY in y0(0): the slow manifold of the fast component is y0 = sin t, zero at t = 0
    struct regime { const char* label; double y00; };
    const regime regimes[2] = {{"on manifold", 0.0}, {"off manifold", 1.0}};
    std::vector<std::vector<double>> err(2), integ(2), iters(2);

    table t({"regime", "scheme", "m", "dt", "k*dt", "best lambda", "rmse", "integ_err", "iters"});
    for (int ri = 0; ri < 2; ++ri) {
        vector_t y0(2);
        y0 << regimes[ri].y00, -0.3;
        // the reference must resolve the fast scale: 100 refinement steps per query point gives k*dt = 0.005
        const eval_grid g(f, y0, T, 401, 100);
        auto study = [&](const char* name, int degree) {
            for (std::size_t gi = 0; gi < ms.size(); ++gi) {
                const int m = ms[gi];
                const vector_t time = vector_t::LinSpaced(m, 0.0, T);
                const stiff_result b = stiff_sweep(degree, f, time, g.at(time, T), g, lambdas, sigma, opt.reps);
                t.row({gi == 0 ? regimes[ri].label : "", gi == 0 ? name : "", num(m), fix(T / (m - 1), 4),
                       fix(k * T / (m - 1), 1), sci(b.lambda, 0), sci(b.err), sci(b.integ), fix(b.iters, 1)});
                err[ri].push_back(b.err);
                integ[ri].push_back(b.integ);
                iters[ri].push_back(b.iters);
            }
            t.rule();
        };
        study("degree 1 (GL1)", 1);
        study("degree 2 (GL2)", 2);
    }
    if (!opt.quiet) {
        std::cout << "  stiffness ratio ~" << k / 2 << ", sigma = " << sigma << ", " << opt.reps
                  << " paired replicates, oracle lambda.\n"
                  << "  on manifold:  y0(0) = 0, the solution is smooth and the stiffness never surfaces.\n"
                  << "  off manifold: y0(0) = 1, a fast layer of width ~" << fix(1.0 / k, 4)
                  << " that no grid here resolves.\n";
        t.print();
    }

    const std::size_t n = ms.size();
    // on the slow manifold, stiffness costs nothing: the fit is noise-limited, not integration-limited
    rep.checkf("on the manifold GL2 is not integration-limited even at large k*dt", integ[0][n] / err[0][n] < 0.2,
               "m=%d, k*dt=%.0f: integ_err %.3e of rmse %.3e", ms[0], k * T / (ms[0] - 1), integ[0][n], err[0][n]);
    // off it, the unresolved layer dominates everything: the error IS the integration error
    rep.checkf("off the manifold the unresolved layer dominates", integ[1][n] / err[1][n] > 0.7,
               "m=%d: integ_err %.3e of rmse %.3e", ms[0], integ[1][n], err[1][n]);
    // and the errors are one to two orders of magnitude larger
    rep.checkf("off-manifold error is an order of magnitude worse", err[1][n] > 5 * err[0][n],
               "m=%d: on %.4e, off %.4e (%.1fx)", ms[0], err[0][n], err[1][n], err[1][n] / err[0][n]);
    // higher order genuinely pays here, unlike everywhere else in this suite
    rep.checkf("off the manifold GL2 clearly beats GL1", err[1][n] < err[1][0],
               "m=%d: GL1 %.4e, GL2 %.4e", ms[0], err[1][0], err[1][n]);
}

/* The control preconditioner's scale assumes the sensitivity of a downstream node to the control is
B_t ~ dt*I. On a stiff component that is wrong by a factor of 1 + O(k*dt): the fast direction is damped
within the step. The scale is a single scalar per interval, so it cannot represent an anisotropy BETWEEN
components at all -- which is why the reduced solve needs many more iterations at small lambda on stiff
problems. This measures the anisotropy directly. */
FDAPDE_BENCHMARK(stiff_control_sensitivity, "control sensitivity B_t against the preconditioner's dt*I assumption") {
    const double k = 200.0;
    const stiff_field f {k};
    vector_t y(2);
    y << 0.3, -0.3;
    table t({"scheme", "dt", "k*dt", "B[stiff]/dt", "B[slow]/dt", "anisotropy"});
    std::vector<double> aniso;
    // NOTE: engine level -- this builds a controlled_ode_solver directly rather than going through
    // bs_ls_ode, so it takes a tableau and is not restricted to the Gauss family the solver requires
    auto study = [&](const char* name, auto tab) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        fdapde::controlled_ode_solver<S, fdapde::Dynamic, stiff_field> engine(
          fdapde::ode_rhs_field<fdapde::Dynamic, stiff_field> {f}, RKIntegrator<S>(tab));
        bool first = true;
        for (double dt : {0.2, 0.1, 0.05, 0.025, 0.0125}) {
            const matrix_t u = matrix_t::Zero(2, S);
            const auto s = engine.step_with_jacobians(0.3, y, dt, u);
            // total sensitivity to a control held constant across the step: sum the per-stage blocks
            double stiff = 0, slow = 0;
            for (int i = 0; i < S; ++i) {
                stiff += s.param(0, i * 2 + 0);
                slow += s.param(1, i * 2 + 1);
            }
            t.row({first ? name : "", fix(dt, 4), fix(k * dt, 1), fix(stiff / dt, 5), fix(slow / dt, 5),
                   fix(slow / stiff, 1)});
            if (std::abs(dt - 0.2) < 1e-12) { aniso.push_back(slow / stiff); }
            first = false;
        }
        t.rule();
    };
    study("implicit_midpoint", ode_schemes::implicit_midpoint());
    study("gauss_legendre_2", ode_schemes::gauss_legendre_2());
    if (!opt.quiet) {
        std::cout << "  the preconditioner assumes B_t ~ dt*I, i.e. both columns equal to 1.0.\n";
        t.print();
    }
    rep.checkf("the stiff direction is damped far below the assumed dt", aniso[0] > 5.0,
               "GL1 at k*dt=40: slow/stiff sensitivity ratio %.1fx", aniso[0]);
    rep.checkf("the anisotropy is worse for the higher-order scheme", aniso[1] > aniso[0],
               "GL1 %.1fx vs GL2 %.1fx", aniso[0], aniso[1]);
}
