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

// Exercises the full-space Gauss-Newton SQP policy of internals::ts_ls_ode (fit_policy::sqp), the
// alternative to the default reduced adjoint-gradient BFGS policy.

#include <cmath>
#include <limits>

using namespace fdapde;

// distinct named namespace: this file is #included into the same translation unit as ts_ode.cpp, so its
// fixtures must not collide with that file's anonymous-namespace helpers.
namespace sqp_ode_test {

using vector_t = Eigen::Matrix<double, Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

constexpr auto SQP = internals::ts_ls_ode::fit_policy::sqp;
constexpr auto ADJ = internals::ts_ls_ode::fit_policy::adjoint;

// nonlinear, non-autonomous test field, d = 2 (same dynamics as the adjoint-solver tests):
//   f(t, y) = [ y0*y1 + sin(t) ; y0 - y1^2 ],   state_jacobian = [ [y1, y0] ; [1, -2*y1] ]
struct nonlinear_field {
    vector_t operator()(double t, const vector_t& y) const {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }
    matrix_t state_jacobian(double, const vector_t& y) const {
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

matrix_t integrate_field(const nonlinear_field& f, const vector_t& time, const vector_t& y0) {
    return RKIntegrator(ode_schemes::gauss_legendre_2()).integrate(ode_rhs_field {f}, time, y0);
}

double rmse(const matrix_t& A, const matrix_t& B) { return std::sqrt((A - B).squaredNorm() / A.size()); }

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
        for (int t = 0; t < m; ++t) {
            Yobs(t, 0) += noise * std::sin(7.0 * t);
            Yobs(t, 1) += noise * std::cos(5.0 * t);
        }
    }
    template <int Stages>
    internals::ts_ls_ode make_solver(const ButcherTableau<Stages>& tab) {
        ts_ls_ode penalty(f, tab);
        internals::ts_ls_ode solver;
        solver.discretize(penalty.get());
        solver.analyze_data(time, Yobs);
        return solver;
    }
    internals::ts_ls_ode make_solver() { return make_solver(ode_schemes::gauss_legendre_2()); }
};

}   // namespace sqp_ode_test

// the SQP-policy solve converges within the iteration budget and the fit denoises.
TEST(ts_ls_ode_sqp, fit_denoises_and_converges) {
    sqp_ode_test::fixture fx;
    auto solver = fx.make_solver();
    solver.fit(1.0, sqp_ode_test::SQP);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(sqp_ode_test::rmse(solver.trajectory(), fx.Ytrue), sqp_ode_test::rmse(fx.Yobs, fx.Ytrue));
}

// lambda spans the data-fit / dynamics trade-off.
TEST(ts_ls_ode_sqp, lambda_controls_tradeoff) {
    sqp_ode_test::fixture fx;
    double defect_prev = std::numeric_limits<double>::infinity();
    double datafit_prev = 0.0;
    for (double lambda : {1e-3, 1e0, 1e3}) {
        auto solver = fx.make_solver();
        solver.fit(lambda, sqp_ode_test::SQP);
        double defect = solver.misfit().cwiseAbs().maxCoeff();
        double datafit = sqp_ode_test::rmse(solver.trajectory(), fx.Yobs);
        EXPECT_LT(defect, defect_prev) << "lambda = " << lambda;
        EXPECT_GT(datafit, datafit_prev) << "lambda = " << lambda;
        defect_prev = defect;
        datafit_prev = datafit;
    }
}

// with noiseless data and large lambda the trajectory is driven onto a discrete ODE solution.
TEST(ts_ls_ode_sqp, large_lambda_enforces_dynamics) {
    sqp_ode_test::fixture fx(41, 2.0, /*noise=*/0.0);
    auto solver = fx.make_solver();
    solver.fit(1e4, sqp_ode_test::SQP);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(solver.misfit().cwiseAbs().maxCoeff(), 1e-3);
    EXPECT_LT(sqp_ode_test::rmse(solver.trajectory(), fx.Ytrue), 1e-2);
}

// a hard initial condition pins the first node exactly.
TEST(ts_ls_ode_sqp, hard_initial_condition) {
    sqp_ode_test::fixture fx;
    vector_t y0(2);
    y0 << 1.234, -0.777;
    ts_ls_ode penalty(fx.f, ode_schemes::gauss_legendre_2(), y0);
    internals::ts_ls_ode solver;
    solver.discretize(penalty.get());
    solver.analyze_data(fx.time, fx.Yobs);
    solver.fit(1.0, sqp_ode_test::SQP);
    EXPECT_NEAR(solver.trajectory()(0, 0), y0[0], 1e-12);
    EXPECT_NEAR(solver.trajectory()(0, 1), y0[1], 1e-12);
}

// missing observations (NaN) are interpolated through; the fit stays finite and converges.
TEST(ts_ls_ode_sqp, handles_missing_observations) {
    sqp_ode_test::fixture fx;
    for (int t = 10; t < 20; ++t) { fx.Yobs(t, 0) = std::numeric_limits<double>::quiet_NaN(); }
    auto solver = fx.make_solver();
    solver.fit(1.0, sqp_ode_test::SQP);
    EXPECT_TRUE(solver.trajectory().allFinite());
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(sqp_ode_test::rmse(solver.trajectory().middleRows(10, 10), fx.Ytrue.middleRows(10, 10)), 0.1);
}

// every time-stepping scheme produces a valid, converged fit under the SQP policy.
TEST(ts_ls_ode_sqp, scheme_variants_converge) {
    sqp_ode_test::fixture fx;
    auto run = [&](auto tab) {
        auto solver = fx.make_solver(tab);
        solver.fit(1.0, sqp_ode_test::SQP);
        EXPECT_TRUE(solver.trajectory().allFinite());
        EXPECT_TRUE(solver.converged());
    };
    run(ode_schemes::forward_euler());
    run(ode_schemes::crank_nicolson());
    run(ode_schemes::implicit_midpoint());
    run(ode_schemes::gauss_legendre_2());
}

// the NPRODE model wrapper drives the SQP-policy fit and still supports GCV.
TEST(ts_ls_ode_sqp, model_wrapper_and_gcv) {
    sqp_ode_test::fixture fx;
    using Model = NPRODE<internals::ts_ls_ode>;
    ts_ls_ode penalty(fx.f, ode_schemes::gauss_legendre_2());
    Model model;
    model.discretize(penalty.get());
    model.analyze_data(fx.time, fx.Yobs);
    model.fit(1.0, sqp_ode_test::SQP);
    EXPECT_TRUE(model.converged());
    EXPECT_EQ(model.n_components(), 2);
    EXPECT_EQ(model.n_nodes(), static_cast<int>(fx.time.size()));
    EXPECT_EQ(model.n_obs(), static_cast<int>(fx.time.size()) * 2);

    // GCV selection is policy-independent (both policies reach the same minimizer); the direct fits below
    // use the SQP policy explicitly.
    std::vector<double> grid;
    for (int e = -4; e <= 3; ++e) { grid.push_back(std::pow(10.0, e)); }
    auto gcv = model.gcv(50, 42);
    GridSearch<1> optimizer;
    optimizer.optimize(gcv, grid);
    double opt = optimizer.optimum()[0];
    EXPECT_GE(opt, grid.front());
    EXPECT_LE(opt, grid.back());
    model.fit(opt, sqp_ode_test::SQP);
    EXPECT_LT(sqp_ode_test::rmse(model.trajectory(), fx.Ytrue), sqp_ode_test::rmse(fx.Yobs, fx.Ytrue));
}

// end-to-end through an order-1 (time) GeoFrame and a formula, fit with the SQP policy on both paths.
TEST(ts_ls_ode_sqp, geoframe_formula_path) {
    sqp_ode_test::fixture fx;
    const int m = fx.time.size();
    Triangulation<1, 1> T = Triangulation<1, 1>::Interval(fx.time[0], fx.time[m - 1], m);
    GeoFrame data(T);
    auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", fx.Yobs);

    ts_ls_ode penalty(fx.f, ode_schemes::gauss_legendre_2());
    NPRODE<internals::ts_ls_ode> model("y ~ f", data, penalty);
    model.fit(1.0, sqp_ode_test::SQP);
    EXPECT_TRUE(model.converged());
    EXPECT_EQ(model.n_nodes(), m);
    EXPECT_EQ(model.n_components(), 2);

    auto solver = fx.make_solver();
    solver.fit(1.0, sqp_ode_test::SQP);
    EXPECT_LT((model.trajectory() - solver.trajectory()).cwiseAbs().maxCoeff(), 1e-9);
}

// effective degrees of freedom lie within the admissible range and decrease with lambda.
TEST(ts_ls_ode_sqp, edf_in_range_and_monotone) {
    sqp_ode_test::fixture fx;
    auto solver = fx.make_solver();
    solver.fit(1e-2, sqp_ode_test::SQP);
    double edf_low_lambda = solver.edf(200, 42);
    solver.fit(1e2, sqp_ode_test::SQP);
    double edf_high_lambda = solver.edf(200, 42);
    const double n_obs = fx.time.size() * 2;
    EXPECT_GT(edf_low_lambda, 0.0);
    EXPECT_LT(edf_low_lambda, n_obs);
    EXPECT_GT(edf_high_lambda, 0.0);
    EXPECT_LT(edf_high_lambda, edf_low_lambda);
}

// the "equal results" guarantee: for every Butcher tableau the two fit policies of the same solver reach
// the same minimizer -- same trajectory, same control, same objective.
TEST(ts_ls_ode_sqp, sqp_policy_matches_adjoint_policy) {
    sqp_ode_test::fixture fx;
    auto check = [&](auto tab) {
        auto adjoint = fx.make_solver(tab);
        adjoint.fit(1.0, sqp_ode_test::ADJ);
        matrix_t Y_adj = adjoint.trajectory();
        matrix_t U_adj = adjoint.control();
        double obj_adj = adjoint.objective();

        auto sqp = fx.make_solver(tab);
        sqp.fit(1.0, sqp_ode_test::SQP);

        EXPECT_TRUE(sqp.converged());
        EXPECT_TRUE(adjoint.converged());
        EXPECT_LT((sqp.trajectory() - Y_adj).cwiseAbs().maxCoeff(), 1e-6);
        EXPECT_LT((sqp.control() - U_adj).cwiseAbs().maxCoeff(), 1e-6);
        EXPECT_NEAR(sqp.objective(), obj_adj, 1e-6 * (1.0 + std::abs(obj_adj)));
    };
    check(ode_schemes::forward_euler());
    check(ode_schemes::crank_nicolson());
    check(ode_schemes::implicit_midpoint());
    check(ode_schemes::gauss_legendre_2());
}
