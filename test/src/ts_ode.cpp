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

#include <cmath>
#include <limits>

using namespace fdapde;

namespace {

using vector_t = Eigen::Matrix<double, Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

// nonlinear, non-autonomous test field, d = 2:
//   f(t, y) = [ y0*y1 + sin(t) ; y0 - y1^2 ],   df_dy = [ [y1, y0] ; [1, -2*y1] ]
struct nonlinear_field {
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }
    matrix_t df_dy(double, const vector_t& y) const {
        matrix_t J(2, 2);
        J << y[1], y[0], 1.0, -2.0 * y[1];
        return J;
    }
};

vector_t make_time(int m, double T) {
    vector_t time(m);
    for (int i = 0; i < m; ++i) { time[i] = T * i / (m - 1); }
    return time;
}

// exact (discrete) trajectory of the prior dynamics from y0, using a high-order scheme
matrix_t integrate_field(const nonlinear_field& f, const vector_t& time, const vector_t& y0) {
    return RKIntegrator(ode_schemes::gauss_legendre_2()).integrate(f, time, y0);
}

double rmse(const matrix_t& A, const matrix_t& B) { return std::sqrt((A - B).squaredNorm() / A.size()); }

// builds a ready-to-fit solver for the nonlinear field over a deterministic noisy data set
struct fixture {
    nonlinear_field f;
    vector_t time;
    matrix_t Ytrue, Yobs;
    fixture(int m = 41, double T = 2.0, double noise = 0.02) {
        time = make_time(m, T);
        vector_t y0(2);
        y0 << 0.5, -0.3;
        Ytrue = integrate_field(f, time, y0);
        Yobs = Ytrue;
        for (int t = 0; t < m; ++t) {   // deterministic, reproducible "noise"
            Yobs(t, 0) += noise * std::sin(7.0 * t);
            Yobs(t, 1) += noise * std::cos(5.0 * t);
        }
    }
    // the solver is templated on neither the stage count nor the system dimension; the tableau and the
    // field carry them, and both are erased away when the penalty descriptor builds its any_rk_engine
    template <int Stages>
    internals::ts_ls_ode make_solver(const ButcherTableau<Stages>& tab) {
        ts_ls_ode penalty(f, tab);   // f returns VectorXd -> Dim deduced Dynamic
        internals::ts_ls_ode solver;
        solver.discretize(penalty.get());
        solver.analyze_data(time, Yobs);
        return solver;
    }
    internals::ts_ls_ode make_solver() { return make_solver(ode_schemes::gauss_legendre_2()); }
};

}   // namespace

// the adjoint-method BFGS solve converges within the iteration budget and the fit denoises
// (closer to truth than the raw data).
TEST(ts_ls_ode, fit_denoises_and_converges) {
    fixture fx;
    auto solver = fx.make_solver();
    solver.fit(1.0);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(rmse(solver.trajectory(), fx.Ytrue), rmse(fx.Yobs, fx.Ytrue));
}

// lambda spans the data-fit / dynamics trade-off: increasing lambda shrinks the control
// defect monotonically and loosens the data fit.
TEST(ts_ls_ode, lambda_controls_tradeoff) {
    fixture fx;
    double defect_prev = std::numeric_limits<double>::infinity();
    double datafit_prev = 0.0;
    for (double lambda : {1e-3, 1e0, 1e3}) {
        auto solver = fx.make_solver();
        solver.fit(lambda);
        double defect = solver.control().cwiseAbs().maxCoeff();
        double datafit = rmse(solver.trajectory(), fx.Yobs);
        EXPECT_LT(defect, defect_prev) << "lambda = " << lambda;
        EXPECT_GT(datafit, datafit_prev) << "lambda = " << lambda;   // looser fit as lambda grows
        defect_prev = defect;
        datafit_prev = datafit;
    }
}

// with noiseless data and large lambda the trajectory is driven onto a discrete ODE solution.
TEST(ts_ls_ode, large_lambda_enforces_dynamics) {
    fixture fx(41, 2.0, /*noise=*/0.0);
    auto solver = fx.make_solver();
    solver.fit(1e4);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(solver.control().cwiseAbs().maxCoeff(), 1e-3);
    EXPECT_LT(rmse(solver.trajectory(), fx.Ytrue), 1e-2);
}

// a hard initial condition pins the first node exactly.
TEST(ts_ls_ode, hard_initial_condition) {
    fixture fx;
    vector_t y0(2);
    y0 << 1.234, -0.777;
    ts_ls_ode penalty(fx.f, ode_schemes::gauss_legendre_2(), y0);
    internals::ts_ls_ode solver;
    solver.discretize(penalty.get());
    solver.analyze_data(fx.time, fx.Yobs);
    solver.fit(1.0);
    EXPECT_NEAR(solver.trajectory()(0, 0), y0[0], 1e-12);
    EXPECT_NEAR(solver.trajectory()(0, 1), y0[1], 1e-12);
}

// missing observations (NaN) are interpolated through; the fit stays finite and converges.
TEST(ts_ls_ode, handles_missing_observations) {
    fixture fx;
    for (int t = 10; t < 20; ++t) { fx.Yobs(t, 0) = std::numeric_limits<double>::quiet_NaN(); }
    auto solver = fx.make_solver();
    solver.fit(1.0);
    EXPECT_TRUE(solver.trajectory().allFinite());
    EXPECT_TRUE(solver.converged());
    // gap region still tracks the truth reasonably
    EXPECT_LT(rmse(solver.trajectory().middleRows(10, 10), fx.Ytrue.middleRows(10, 10)), 0.1);
}

// every time-stepping scheme produces a valid, converged fit.
TEST(ts_ls_ode, scheme_variants_converge) {
    fixture fx;
    // the schemes have distinct stage counts (ButcherTableau<Stages>), so each is exercised through a
    // generic lambda rather than a single heterogeneous loop.
    auto run = [&](auto tab) {
        auto solver = fx.make_solver(tab);
        solver.fit(1.0);
        EXPECT_TRUE(solver.trajectory().allFinite());
        EXPECT_TRUE(solver.converged());
    };
    run(ode_schemes::forward_euler());
    run(ode_schemes::crank_nicolson());
    run(ode_schemes::implicit_midpoint());
    run(ode_schemes::gauss_legendre_2());
}

// the NPRODE model wrapper drives the solver and selects lambda by GCV over a grid.
TEST(ts_ls_ode, model_wrapper_and_gcv) {
    fixture fx;
    using Model = NPRODE<internals::ts_ls_ode>;
    ts_ls_ode penalty(fx.f, ode_schemes::gauss_legendre_2());
    Model model;
    model.discretize(penalty.get());
    model.analyze_data(fx.time, fx.Yobs);
    model.fit(1.0);
    EXPECT_TRUE(model.converged());
    EXPECT_EQ(model.n_components(), 2);
    EXPECT_EQ(model.n_nodes(), static_cast<int>(fx.time.size()));
    EXPECT_EQ(model.n_obs(), static_cast<int>(fx.time.size()) * 2);

    std::vector<double> grid;
    for (int e = -4; e <= 3; ++e) { grid.push_back(std::pow(10.0, e)); }
    auto gcv = model.gcv(50, 42);
    GridSearch<1> optimizer;
    optimizer.optimize(gcv, grid);
    double opt = optimizer.optimum()[0];
    EXPECT_GE(opt, grid.front());
    EXPECT_LE(opt, grid.back());
    model.fit(opt);
    EXPECT_LT(rmse(model.trajectory(), fx.Ytrue), rmse(fx.Yobs, fx.Ytrue));   // GCV choice denoises
}

// end-to-end through an order-1 (time) GeoFrame and a formula: the multi-column response is
// read as a block column and the fit matches the plain-input path exactly.
TEST(ts_ls_ode, geoframe_formula_path) {
    fixture fx;
    const int m = fx.time.size();
    // order-1 time mesh matching fx.time, with the d-dimensional response as a block column
    Triangulation<1, 1> T = Triangulation<1, 1>::Interval(fx.time[0], fx.time[m - 1], m);
    GeoFrame data(T);
    auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", fx.Yobs);

    ts_ls_ode penalty(fx.f, ode_schemes::gauss_legendre_2());
    NPRODE<internals::ts_ls_ode> model("y ~ f", data, penalty);
    model.fit(1.0);
    EXPECT_TRUE(model.converged());
    EXPECT_EQ(model.n_nodes(), m);
    EXPECT_EQ(model.n_components(), 2);

    // identical to fitting through the plain (time, y_obs) interface
    auto solver = fx.make_solver();
    solver.fit(1.0);
    EXPECT_LT((model.trajectory() - solver.trajectory()).cwiseAbs().maxCoeff(), 1e-9);
}

// effective degrees of freedom lie within the admissible range and decrease with lambda.
TEST(ts_ls_ode, edf_in_range_and_monotone) {
    fixture fx;
    auto solver = fx.make_solver();
    solver.fit(1e-2);
    double edf_low_lambda = solver.edf(200, 42);
    solver.fit(1e2);
    double edf_high_lambda = solver.edf(200, 42);
    const double n_obs = fx.time.size() * 2;
    EXPECT_GT(edf_low_lambda, 0.0);
    EXPECT_LT(edf_low_lambda, n_obs);
    EXPECT_GT(edf_high_lambda, 0.0);
    EXPECT_LT(edf_high_lambda, edf_low_lambda);   // more regularization -> fewer effective dof
}

// the discrete adjoint of one forward step reproduces the finite-difference sensitivities of the
// step w.r.t. the initial state and the additive control, for every Butcher tableau. This is the
// core correctness check behind RKIntegrator::adjoint_step (hence the reduced-problem gradient).
TEST(ts_ls_ode, adjoint_step_matches_finite_differences) {
    nonlinear_field f;
    const int d = 2;
    vector_t y(d), u(d), w(d);
    y << 0.4, -0.2;
    u << 0.1, -0.05;
    w << 1.3, -0.7;   // linear cost C = w . y_next, so the incoming costate is p_next = w
    const double t = 0.3, dt = 0.05, fd = 1e-6;
    // exercised per scheme through a generic lambda: the tableaux have distinct stage counts and so
    // cannot share a single heterogeneous loop.
    auto check = [&](auto tab) {
        RKIntegrator integ(tab);
        ode_rhs_field field(f);
        auto step = [&](const vector_t& yy, const vector_t& uu) {
            return integ.step(field + uu, t, yy, dt);
        };
        auto [p_curr, grad_contrib] = integ.adjoint_step(field + u, t, y, dt, w);
        for (int j = 0; j < d; ++j) {
            vector_t yp = y, ym = y, up = u, um = u;
            yp[j] += fd, ym[j] -= fd, up[j] += fd, um[j] -= fd;
            double dC_dy = w.dot(step(yp, u) - step(ym, u)) / (2 * fd);   // d(w.y_next)/dy_j
            double dC_du = w.dot(step(y, up) - step(y, um)) / (2 * fd);   // d(w.y_next)/du_j
            EXPECT_NEAR(p_curr[j], dC_dy, 1e-5) << "dC/dy, j = " << j;
            EXPECT_NEAR(grad_contrib[j], dC_du, 1e-5) << "dC/du, j = " << j;
        }
    };
    check(ode_schemes::forward_euler());
    check(ode_schemes::crank_nicolson());
    check(ode_schemes::implicit_midpoint());
    check(ode_schemes::gauss_legendre_2());
}

// the same dynamics as nonlinear_field but WITHOUT an analytic df_dy: routes ode_rhs_field through the
// MatrixField-based finite-difference Jacobian. Returning VectorXd keeps it on the dynamic-Dim path.
struct nonlinear_field_no_jac {
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }
};

// the same dynamics with a fixed-size (Vector2d) return: ode_rhs_field deduces a static Dim = 2.
struct nonlinear_field_static {
    Eigen::Vector2d operator()(double t, const vector_t& y) const {
        Eigen::Vector2d out;
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }
    Eigen::Matrix2d df_dy(double, const vector_t& y) const {
        Eigen::Matrix2d J;
        J << y[1], y[0], 1.0, -2.0 * y[1];
        return J;
    }
};

// an ode_rhs_field built from a callable WITHOUT an analytic Jacobian falls back to the MatrixField
// finite-difference Jacobian; it must agree with the analytic Jacobian of the same dynamics. Also
// checks the value path and that field + c offsets the value while preserving the (FD) Jacobian.
TEST(ts_ls_ode, ode_rhs_field_fd_jacobian_matches_analytic) {
    nonlinear_field f;                 // provides an analytic df_dy
    ode_rhs_field fd_field{nonlinear_field_no_jac{}};   // no df_dy -> FD Jacobian backend
    vector_t y(2), c(2);
    y << 0.4, -0.2;
    c << 0.5, -0.25;
    const double t = 0.3;
    EXPECT_LT((fd_field.df_dy(t, y) - f.df_dy(t, y)).cwiseAbs().maxCoeff(), 1e-4);   // FD Jacobian
    EXPECT_LT((fd_field(t, y) - f(t, y)).cwiseAbs().maxCoeff(), 1e-12);              // value path
    ode_rhs_field shifted = fd_field + c;
    EXPECT_LT((shifted.df_dy(t, y) - f.df_dy(t, y)).cwiseAbs().maxCoeff(), 1e-4);    // Jacobian unchanged by +c
    EXPECT_LT((shifted(t, y) - (f(t, y) + c)).cwiseAbs().maxCoeff(), 1e-12);         // value offset by c
}

// a field with a fixed-size return makes the stage math static: Dim is deduced (no n_components()),
// ode_rhs_field<2> and an internals::rk_engine<Stages, 2> are built inside the (non-templated) solver,
// and the fit (through the fixed-size RKIntegrator stage math) denoises just like the dynamic path.
TEST(ts_ls_ode, static_dim_deduced_and_fits) {
    nonlinear_field_static sf;
    ode_rhs_field field{sf};                                  // Dim deduced from the Vector2d return
    static_assert(decltype(field)::dim == 2, "static dimension must be deduced from the field");
    vector_t y(2), c(2);
    y << 0.4, -0.2;
    c << 0.5, -0.25;
    // field algebra and Jacobian still hold on the static field
    EXPECT_LT(((field + c)(0.3, y) - (field(0.3, y) + c)).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT(((field + c).df_dy(0.3, y) - field.df_dy(0.3, y)).cwiseAbs().maxCoeff(), 1e-12);

    // end-to-end fit through the static-dimension stack
    fixture fx;
    ts_ls_ode penalty(sf, ode_schemes::gauss_legendre_2());   // Dim = 2 deduced internally, not in the type
    static_assert(decltype(penalty)::solver_t::n_lambda == 1);   // solver type is dimension-agnostic now
    internals::ts_ls_ode solver;
    solver.discretize(penalty.get());
    solver.analyze_data(fx.time, fx.Yobs);
    solver.fit(1.0);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(rmse(solver.trajectory(), fx.Ytrue), rmse(fx.Yobs, fx.Ytrue));
}
