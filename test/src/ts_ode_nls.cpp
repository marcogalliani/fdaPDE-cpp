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

// Exercises the nonlinear least squares (single shooting) inverse solver (internals::ts_ls_ode_nls):
// ODE parameter estimation with S(theta, y0) = || y(theta, y0) - y_obs ||^2 and the exact discrete
// adjoint gradient of the shooting map. The AD-guarded block at the bottom additionally checks the
// autodiff adapter (fdaPDE/autodiff.h): the same model written once, generic in the scalar type, must
// reproduce the hand-differentiated field exactly.

#include <cmath>
#include <limits>

using namespace fdapde;

namespace nls_ode_test {

using vector_t = Eigen::Matrix<double, Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
using nls_solver = internals::ts_ls_ode_nls;

// the same theta-parameterized field the tracking-solver tests use, so the two estimators are exercised
// on identical dynamics: f(t, y, th) = [ th0*y0*y1 + sin(t) ; th1*y0 - th2*y1^2 ], d = 2, n_theta = 3
struct param_field {
    vector_t operator()(double t, const vector_t& y, const vector_t& th) const {
        vector_t out(2);
        out << th[0] * y[0] * y[1] + std::sin(t), th[1] * y[0] - th[2] * y[1] * y[1];
        return out;
    }
    matrix_t state_jacobian(double, const vector_t& y, const vector_t& th) const {
        matrix_t J(2, 2);
        J << th[0] * y[1], th[0] * y[0], th[1], -2.0 * th[2] * y[1];
        return J;
    }
    matrix_t param_jacobian(double, const vector_t& y, const vector_t& th) const {
        matrix_t J = matrix_t::Zero(2, 3);
        J(0, 0) = y[0] * y[1];
        J(1, 1) = y[0];
        J(1, 2) = -y[1] * y[1];
        (void)th;
        return J;
    }
};

vector_t make_time(int m, double T) {
    vector_t time(m);
    for (int i = 0; i < m; ++i) { time[i] = T * i / (m - 1); }
    return time;
}
vector_t theta_of(double a, double b, double c) {
    vector_t th(3);
    th << a, b, c;
    return th;
}
matrix_t integrate_param(const vector_t& theta, const vector_t& time, const vector_t& y0) {
    ode_rhs_field field {param_field {}, theta};
    return RKIntegrator(ode_schemes::gauss_legendre_2()).integrate(field, time, y0);
}

struct fixture {
    vector_t theta_true, time, y0;
    matrix_t Ytrue, Yobs;
    fixture(int m = 31, double T = 2.0, double noise = 0.0) {
        theta_true = theta_of(1.0, 1.0, 1.0);
        time = make_time(m, T);
        y0.resize(2);
        y0 << 0.5, -0.3;
        Ytrue = integrate_param(theta_true, time, y0);
        Yobs = Ytrue;
        for (int t = 0; t < m; ++t) {   // deterministic, reproducible "noise"
            Yobs(t, 0) += noise * std::sin(7.0 * t);
            Yobs(t, 1) += noise * std::cos(5.0 * t);
        }
    }
    // solver with the initial state PINNED at the truth (theta is then the only unknown)
    template <typename Field, int Stages>
    nls_solver make_solver(Field field, const ButcherTableau<Stages>& tab, const vector_t& theta0) {
        ts_ls_ode_param descriptor(field, tab, theta0, y0, /*max_iter=*/200, /*tol=*/1e-10);
        nls_solver solver;
        solver.discretize(descriptor.get());
        solver.analyze_data(time, Yobs);
        return solver;
    }
    nls_solver make_solver(const vector_t& theta0) {
        return make_solver(param_field {}, ode_schemes::gauss_legendre_2(), theta0);
    }
    // solver with a FREE initial state (estimated jointly with theta)
    nls_solver make_free_ic_solver(const vector_t& theta0) {
        ts_ls_ode_param descriptor(param_field {}, ode_schemes::gauss_legendre_2(), theta0, 200, 1e-10);
        nls_solver solver;
        solver.discretize(descriptor.get());
        solver.analyze_data(time, Yobs);
        return solver;
    }
};

// central-difference check of the analytic gradient at z, componentwise
void expect_gradient_matches_fd(nls_solver& solver, const vector_t& z, double tol, const char* what) {
    vector_t g = solver.gradient_at(z);
    ASSERT_EQ(g.size(), z.size());
    const double h = 1e-6;
    for (int k = 0; k < z.size(); ++k) {
        vector_t zp = z, zm = z;
        zp[k] += h;
        zm[k] -= h;
        double g_fd = (solver.objective_at(zp) - solver.objective_at(zm)) / (2 * h);
        EXPECT_NEAR(g[k], g_fd, tol * (1.0 + std::abs(g_fd))) << what << ", k = " << k;
    }
}

}   // namespace nls_ode_test

// THE key correctness test: the discrete adjoint gradient of the shooting map reproduces central
// finite differences of S(theta), for every Butcher tableau (an implicit tableau must differentiate
// through the Newton stage solve, which is where a wrong sensitivity would hide).
TEST(ts_ls_ode_nls, gradient_matches_finite_differences) {
    nls_ode_test::fixture fx(21, 2.0, /*noise=*/0.01);
    vector_t th = nls_ode_test::theta_of(0.85, 1.15, 0.9);   // away from the optimum
    auto check = [&](auto tab, const char* name) {
        auto solver = fx.make_solver(nls_ode_test::param_field {}, tab, th);
        nls_ode_test::expect_gradient_matches_fd(solver, th, 1e-5, name);
    };
    check(ode_schemes::forward_euler(), "forward_euler");
    check(ode_schemes::crank_nicolson(), "crank_nicolson");
    check(ode_schemes::implicit_midpoint(), "implicit_midpoint");
    check(ode_schemes::gauss_legendre_2(), "gauss_legendre_2");
}

// the parameter sensitivity path also works when df/dtheta is not analytic: ode_rhs_field's central
// differences feed the same adjoint sweep
TEST(ts_ls_ode_nls, gradient_with_fd_param_jacobian) {
    struct field_no_dtheta {   // same dynamics, no param_jacobian
        vector_t operator()(double t, const vector_t& y, const vector_t& th) const {
            return nls_ode_test::param_field {}(t, y, th);
        }
        matrix_t state_jacobian(double t, const vector_t& y, const vector_t& th) const {
            return nls_ode_test::param_field {}.state_jacobian(t, y, th);
        }
    };
    nls_ode_test::fixture fx(21, 2.0, /*noise=*/0.01);
    vector_t th = nls_ode_test::theta_of(0.85, 1.15, 0.9);
    auto solver_a = fx.make_solver(th);
    auto solver_b = fx.make_solver(field_no_dtheta {}, ode_schemes::gauss_legendre_2(), th);
    vector_t g_analytic = solver_a.gradient_at(th), g_fd = solver_b.gradient_at(th);
    EXPECT_LT((g_analytic - g_fd).cwiseAbs().maxCoeff(), 1e-5 * (1.0 + g_analytic.cwiseAbs().maxCoeff()));
}

// end-to-end: recover the true parameters from (noiseless) data generated by the ODE itself
TEST(ts_ls_ode_nls, recovers_true_parameters) {
    nls_ode_test::fixture fx(31, 2.0, /*noise=*/0.0);
    vector_t theta0 = nls_ode_test::theta_of(0.7, 1.3, 0.8);
    auto solver = fx.make_solver(theta0);
    solver.set_options(200, 1e-10);
    solver.solve(theta0);
    EXPECT_LT((solver.theta() - fx.theta_true).cwiseAbs().maxCoeff(), 1e-4);
    EXPECT_LT(solver.outer_objective(), 1e-12);
    // the forward state left behind is the shooting solution at the estimate
    EXPECT_LT((solver.trajectory() - fx.Ytrue).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_EQ(solver.n_params(), 3);
    EXPECT_EQ(solver.n_unknowns(), 3);   // hard IC -> theta only
}

// noise degrades the estimate gracefully: less noise -> closer to the truth
TEST(ts_ls_ode_nls, estimate_improves_as_noise_vanishes) {
    vector_t theta0 = nls_ode_test::theta_of(0.8, 1.2, 0.9);
    double err_prev = std::numeric_limits<double>::infinity();
    for (double noise : {2e-2, 2e-3, 0.0}) {
        nls_ode_test::fixture fx(31, 2.0, noise);
        auto solver = fx.make_solver(theta0);
        solver.set_options(200, 1e-10);
        solver.solve(theta0);
        double err = (solver.theta() - fx.theta_true).cwiseAbs().maxCoeff();
        EXPECT_LT(err, err_prev) << "noise = " << noise;
        err_prev = err;
    }
}

// a free initial state becomes a decision variable: the gradient block is the node-0 costate, and both
// theta and y0 are recovered from noiseless data
TEST(ts_ls_ode_nls, free_initial_state_is_estimated_jointly) {
    nls_ode_test::fixture fx(31, 2.0, /*noise=*/0.0);
    vector_t theta0 = nls_ode_test::theta_of(0.8, 1.2, 0.9);
    auto solver = fx.make_free_ic_solver(theta0);
    EXPECT_TRUE(solver.estimates_ic());
    EXPECT_EQ(solver.n_unknowns(), 5);   // 3 parameters + a 2-dimensional initial state
    vector_t z = solver.decision_vector();
    z.head(3) = theta0;
    nls_ode_test::expect_gradient_matches_fd(solver, z, 1e-5, "free initial state");
    solver.set_options(300, 1e-10);
    solver.solve(theta0);
    EXPECT_LT((solver.theta() - fx.theta_true).cwiseAbs().maxCoeff(), 1e-3);
    EXPECT_LT((solver.initial_state() - fx.y0).cwiseAbs().maxCoeff(), 1e-3);
}

// a hard initial condition pins the first node and is not estimated
TEST(ts_ls_ode_nls, hard_initial_condition) {
    nls_ode_test::fixture fx(21, 2.0, /*noise=*/0.01);
    vector_t theta0 = nls_ode_test::theta_of(0.9, 1.1, 0.95);
    auto solver = fx.make_solver(theta0);
    EXPECT_FALSE(solver.estimates_ic());
    solver.solve(theta0);
    EXPECT_NEAR(solver.trajectory()(0, 0), fx.y0[0], 1e-12);
    EXPECT_NEAR(solver.trajectory()(0, 1), fx.y0[1], 1e-12);
    EXPECT_LT((solver.theta() - fx.theta_true).cwiseAbs().maxCoeff(), 5e-2);
}

// an initial condition that is itself a function of theta contributes through d(y0)/d(theta)
TEST(ts_ls_ode_nls, ic_parameterization) {
    nls_ode_test::fixture fx(21, 2.0, /*noise=*/0.01);
    vector_t theta0 = nls_ode_test::theta_of(0.9, 1.1, 0.95);
    auto solver = fx.make_free_ic_solver(theta0);
    // y0(theta) = (0.5 * th0, -0.3 * th1): at theta = (1,1,1) it reproduces the true initial state
    solver.set_ic_parameterization(
      [](const vector_t& th) {
          vector_t y0(2);
          y0 << 0.5 * th[0], -0.3 * th[1];
          return y0;
      },
      [](const vector_t&) {
          matrix_t J = matrix_t::Zero(2, 3);
          J(0, 0) = 0.5;
          J(1, 1) = -0.3;
          return J;
      });
    EXPECT_FALSE(solver.estimates_ic());
    EXPECT_EQ(solver.n_unknowns(), 3);
    nls_ode_test::expect_gradient_matches_fd(solver, theta0, 1e-5, "theta-parameterized ic");
    solver.set_options(300, 1e-10);
    solver.solve(theta0);
    // the initial state follows the estimate through the parameterization
    EXPECT_NEAR(solver.trajectory()(0, 0), 0.5 * solver.theta()[0], 1e-12);
    EXPECT_NEAR(solver.trajectory()(0, 1), -0.3 * solver.theta()[1], 1e-12);
}

// missing observations (NaN) are excluded from the criterion and from its gradient
TEST(ts_ls_ode_nls, missing_observations) {
    nls_ode_test::fixture fx(21, 2.0, /*noise=*/0.01);
    vector_t theta0 = nls_ode_test::theta_of(0.9, 1.1, 0.95);
    matrix_t Y = fx.Yobs;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int t = 3; t < 8; ++t) { Y(t, 1) = nan; }   // a gap in the second component
    ts_ls_ode_param descriptor(nls_ode_test::param_field {}, ode_schemes::gauss_legendre_2(), theta0, fx.y0, 200, 1e-10);
    internals::ts_ls_ode_nls solver;
    solver.discretize(descriptor.get());
    solver.analyze_data(fx.time, Y);
    EXPECT_EQ(solver.n_obs(), 21 * 2 - 5);
    EXPECT_TRUE(std::isfinite(solver.objective_at(theta0)));
    nls_ode_test::expect_gradient_matches_fd(solver, theta0, 1e-5, "missing observations");
    solver.set_options(200, 1e-10);
    solver.solve(theta0);
    EXPECT_LT((solver.theta() - fx.theta_true).cwiseAbs().maxCoeff(), 5e-2);
}

// a parameter whose forward integration blows up must yield a large but FINITE cost, and a gradient that
// still points somewhere: the criterion is scored on the finite prefix of the trajectory, so the search
// can walk back into the stable region instead of dying on a NaN
TEST(ts_ls_ode_nls, divergent_parameters_do_not_break_the_solve) {
    nls_ode_test::fixture fx(21, 2.0, /*noise=*/0.0);
    vector_t wild = nls_ode_test::theta_of(1e3, 1e3, 1e3);   // explosive dynamics
    auto solver = fx.make_solver(nls_ode_test::param_field {}, ode_schemes::forward_euler(), wild);
    double S = solver.objective_at(wild);
    EXPECT_TRUE(std::isfinite(S));
    EXPECT_GT(S, 1e12);                              // dominates any attainable finite value
    EXPECT_TRUE(solver.gradient_at(wild).allFinite());
    EXPECT_GT(solver.n_divergent_evals(), 0);
    // and the solve still returns a usable (finite) estimate rather than NaN
    solver.set_options(50, 1e-8);
    solver.solve(wild);
    EXPECT_TRUE(solver.theta().allFinite());
}

// the NLS solver is a ts_ls_ode: data ingestion, the observation mask and the forward-state observers are
// the base's, so the estimate is reported through the same API as the tracking solver's
TEST(ts_ls_ode_nls, mirrors_the_forward_solver_api) {
    nls_ode_test::fixture fx(31, 2.0, /*noise=*/0.01);
    vector_t theta0 = nls_ode_test::theta_of(0.9, 1.1, 0.95);
    auto solver = fx.make_solver(theta0);
    solver.solve(theta0);
    EXPECT_EQ(solver.n_components(), 2);
    EXPECT_EQ(solver.n_nodes(), 31);
    EXPECT_EQ(solver.n_obs(), 62);
    EXPECT_TRUE(solver.trajectory().allFinite());
    EXPECT_EQ(solver.trajectory().rows(), 31);
    EXPECT_EQ(solver.control().rows(), 30);
    EXPECT_TRUE(solver.control().isZero(0.0));   // NLS stays on the model manifold: no control
    EXPECT_NEAR(solver.rss(), solver.outer_objective(), 1e-10 * (1 + solver.rss()));
    EXPECT_EQ(solver.n_iter(), solver.outer_n_iter());
}

// the two line-search policies solve the same problem; both must land on the same optimum from a start
// close enough that neither struggles
TEST(ts_ls_ode_nls, line_search_agreement) {
    nls_ode_test::fixture fx(31, 2.0, /*noise=*/0.01);
    vector_t theta0 = nls_ode_test::theta_of(0.95, 1.05, 0.98);
    auto solver_b = fx.make_solver(theta0);
    solver_b.set_line_search(internals::ts_ls_ode_nls::line_search::backtracking);
    solver_b.set_options(300, 1e-10);
    solver_b.solve(theta0);
    auto solver_w = fx.make_solver(theta0);
    solver_w.set_line_search(internals::ts_ls_ode_nls::line_search::wolfe);
    solver_w.set_options(300, 1e-10);
    solver_w.solve(theta0);
    EXPECT_LT((solver_b.theta() - solver_w.theta()).cwiseAbs().maxCoeff(), 1e-3);
}

#ifdef FDAPDE_HAS_AUTODIFF

namespace nls_ode_test {

// the same dynamics as param_field, written ONCE and generic in the scalar type: no hand-written
// Jacobians. ad_ode_rhs derives both of them by forward-mode automatic differentiation.
struct rosenbrock {
    template <typename Scalar> Scalar operator()(const Eigen::Matrix<Scalar, Dynamic, 1>& x) const {
        return (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
    }
};

struct generic_field {
    template <typename Scalar>
    Eigen::Matrix<Scalar, 2, 1> operator()(
      double t, const Eigen::Matrix<Scalar, Dynamic, 1>& y, const Eigen::Matrix<Scalar, Dynamic, 1>& th) const {
        Eigen::Matrix<Scalar, 2, 1> out;
        out[0] = th[0] * y[0] * y[1] + std::sin(t);
        out[1] = th[1] * y[0] - th[2] * y[1] * y[1];
        return out;
    }
};

}   // namespace nls_ode_test

// the AD-derived Jacobians reproduce the hand-written ones to machine precision, and the wrapper keeps
// the field on the integrator's static-dimension path
TEST(ad_ode_rhs, jacobians_match_the_analytic_field) {
    using ad_field_t = ad_ode_rhs<nls_ode_test::generic_field>;
    static_assert(ode_rhs_dim_v<ad_field_t> == 2, "a fixed-size return must survive the adapter");
    static_assert(is_parameterized_ode_rhs<ad_field_t>);
    static_assert(parameterized_ode_rhs_has_state_jacobian<ad_field_t>);
    static_assert(parameterized_ode_rhs_has_param_jacobian<ad_field_t>);
    ad_field_t ad {nls_ode_test::generic_field {}};
    nls_ode_test::param_field analytic;
    vector_t y(2), th(3);
    y << 0.4, -0.2;
    th << 0.8, 1.2, 0.9;
    const double t = 0.3;
    EXPECT_LT((vector_t(ad(t, y, th)) - analytic(t, y, th)).cwiseAbs().maxCoeff(), 1e-14);
    EXPECT_LT((ad.state_jacobian(t, y, th) - analytic.state_jacobian(t, y, th)).cwiseAbs().maxCoeff(), 1e-14);
    EXPECT_LT((ad.param_jacobian(t, y, th) - analytic.param_jacobian(t, y, th)).cwiseAbs().maxCoeff(), 1e-14);
}

// end-to-end: an AD-differentiated field and a hand-differentiated one are the same estimator -- same
// gradient, same estimate
TEST(ts_ls_ode_nls, autodiff_field_matches_analytic_field) {
    nls_ode_test::fixture fx(31, 2.0, /*noise=*/0.01);
    vector_t theta0 = nls_ode_test::theta_of(0.8, 1.2, 0.9);
    auto solver_ad = fx.make_solver(
      make_ad_ode_rhs(nls_ode_test::generic_field {}), ode_schemes::gauss_legendre_2(), theta0);
    auto solver_an = fx.make_solver(theta0);
    EXPECT_LT((solver_ad.gradient_at(theta0) - solver_an.gradient_at(theta0)).cwiseAbs().maxCoeff(), 1e-9);
    solver_ad.set_options(200, 1e-10);
    solver_an.set_options(200, 1e-10);
    solver_ad.solve(theta0);
    solver_an.solve(theta0);
    EXPECT_LT((solver_ad.theta() - solver_an.theta()).cwiseAbs().maxCoeff(), 1e-6);
}

// the AD objective adapter satisfies the interface the optimization module expects, so any gradient-based
// algorithm can optimize a value-only objective
TEST(ad_objective, drives_the_optimization_module) {
    auto objective = make_ad_objective(nls_ode_test::rosenbrock {});
    vector_t x0(2);
    x0 << -1.2, 1.0;
    // the AD gradient against central differences
    auto grad = objective.gradient();
    vector_t g = grad(x0);
    const double h = 1e-6;
    for (int k = 0; k < 2; ++k) {
        vector_t xp = x0, xm = x0;
        xp[k] += h;
        xm[k] -= h;
        EXPECT_NEAR(g[k], (objective(xp) - objective(xm)) / (2 * h), 1e-4 * (1 + std::abs(g[k])));
    }
    BFGS<Dynamic> bfgs(500, 1e-8, 1.0);
    vector_t x_bfgs = bfgs.optimize(objective, x0, WolfeLineSearch());
    EXPECT_LT((x_bfgs - vector_t::Ones(2)).cwiseAbs().maxCoeff(), 1e-4);
    Newton<Dynamic> newton(200, 1e-8, 1.0);   // also exercises the AD Hessian
    vector_t x_newton = newton.optimize(objective, x0, BacktrackingLineSearch());
    EXPECT_LT((x_newton - vector_t::Ones(2)).cwiseAbs().maxCoeff(), 1e-4);
}

#endif   // FDAPDE_HAS_AUTODIFF
