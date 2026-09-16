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

// a constant additive forcing c with zero state Jacobian, in the system dimension Dim
template <int Dim> struct constant_forcing {
    static constexpr int dim = Dim;
    vector_t c;
    vector_t operator()(double, const vector_t&) const { return c; }
    matrix_t state_jacobian(double, const vector_t& y) const { return matrix_t::Zero(y.size(), y.size()); }
};
// f + c: the prior field forced by a constant control c, formed via ode_rhs_field's field-field addition. Dim is
// deduced from f so the addition type-checks.
template <int Dim, typename F>
auto operator_plus_c(const ode_rhs_field<Dim, F>& f, const vector_t& c) {
    return f + ode_rhs_field<Dim, constant_forcing<Dim>>(constant_forcing<Dim> {c});
}

// nonlinear, non-autonomous test field, d = 2:
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

// exact (discrete) trajectory of the prior dynamics from y0, using a high-order scheme
matrix_t integrate_field(const nonlinear_field& f, const vector_t& time, const vector_t& y0) {
    return RKIntegrator(ode_schemes::gauss_legendre_2()).integrate(ode_rhs_field {f}, time, y0);
}

double rmse(const matrix_t& A, const matrix_t& B) { return std::sqrt((A - B).squaredNorm() / A.size()); }

// builds a ready-to-fit solver for the nonlinear field over a deterministic noisy data set
struct fixture {
    nonlinear_field f;
    vector_t time;
    Triangulation<1, 1> mesh;   // outlives every solver built here (see make_solver)
    matrix_t Ytrue, Yobs;
    fixture(int m = 41, double T = 2.0, double noise = 0.02) {
        time = make_time(m, T);
        mesh = Triangulation<1, 1>(time);
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
    // field carry them, and both are erased away when the penalty descriptor builds its any_controlled_ode_solver
    /* The trajectory space is the only discretization choice: its degree p selects the p-stage Gauss
    scheme. The solver reads the space in discretize and keeps nothing of it, so it is local here; the mesh
    must outlive the solver. */
    internals::bs_ls_ode make_solver(int degree) {
        BsSpace<Triangulation<1, 1>> Vh(mesh, degree, std::vector<int>(time.size(), degree));   // C0 at every node
        bs_ls_ode penalty(f, Vh);   // f returns VectorXd -> Dim deduced Dynamic
        internals::bs_ls_ode solver;
        solver.discretize(penalty.get());
        solver.analyze_data(time, Yobs);
        return solver;
    }
    internals::bs_ls_ode make_solver() { return make_solver(2); }
};

// the Gauss-Legendre family, the only schemes the solver admits. The tableaux have distinct types, so the
// visitor is generic and invoked once per scheme.
template <typename Visitor> void for_each_gauss_scheme(Visitor&& visit) {
    visit("implicit_midpoint", ode_schemes::implicit_midpoint());
    visit("gauss_legendre_2",  ode_schemes::gauss_legendre_2());
    visit("gauss_legendre_3",  ode_schemes::gauss_legendre_3());
    visit("gauss_legendre_4",  ode_schemes::gauss_legendre_4());
}

}   // namespace

// the adjoint-method BFGS solve converges within the iteration budget and the fit denoises
// (closer to truth than the raw data).
TEST(bs_ls_ode, fit_denoises_and_converges) {
    fixture fx;
    auto solver = fx.make_solver();
    solver.fit(1.0);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(rmse(solver.trajectory(), fx.Ytrue), rmse(fx.Yobs, fx.Ytrue));
}

// lambda spans the data-fit / dynamics trade-off: increasing lambda shrinks the control
// defect monotonically and loosens the data fit.
TEST(bs_ls_ode, lambda_controls_tradeoff) {
    fixture fx;
    double defect_prev = std::numeric_limits<double>::infinity();
    double datafit_prev = 0.0;
    for (double lambda : {1e-3, 1e0, 1e3}) {
        auto solver = fx.make_solver();
        solver.fit(lambda);
        double defect = solver.misfit().cwiseAbs().maxCoeff();
        double datafit = rmse(solver.trajectory(), fx.Yobs);
        EXPECT_LT(defect, defect_prev) << "lambda = " << lambda;
        EXPECT_GT(datafit, datafit_prev) << "lambda = " << lambda;   // looser fit as lambda grows
        defect_prev = defect;
        datafit_prev = datafit;
    }
}

// with noiseless data and large lambda the trajectory is driven onto a discrete ODE solution.
TEST(bs_ls_ode, large_lambda_enforces_dynamics) {
    fixture fx(41, 2.0, /*noise=*/0.0);
    auto solver = fx.make_solver();
    solver.fit(1e4);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(solver.misfit().cwiseAbs().maxCoeff(), 1e-3);
    EXPECT_LT(rmse(solver.trajectory(), fx.Ytrue), 1e-2);
}

// a hard initial condition pins the first node exactly.
TEST(bs_ls_ode, hard_initial_condition) {
    fixture fx;
    vector_t y0(2);
    y0 << 1.234, -0.777;
    Triangulation<1, 1> T(fx.time);
    BsSpace Vh(T, 2, std::vector<int>(T.n_nodes(), 2));   // C0 at every node
    bs_ls_ode penalty(fx.f, Vh, y0);
    internals::bs_ls_ode solver;
    solver.discretize(penalty.get());
    solver.analyze_data(fx.time, fx.Yobs);
    solver.fit(1.0);
    EXPECT_NEAR(solver.trajectory()(0, 0), y0[0], 1e-12);
    EXPECT_NEAR(solver.trajectory()(0, 1), y0[1], 1e-12);
}

// missing observations (NaN) are interpolated through; the fit stays finite and converges.
TEST(bs_ls_ode, handles_missing_observations) {
    fixture fx;
    for (int t = 10; t < 20; ++t) { fx.Yobs(t, 0) = std::numeric_limits<double>::quiet_NaN(); }
    auto solver = fx.make_solver();
    solver.fit(1.0);
    EXPECT_TRUE(solver.trajectory().allFinite());
    EXPECT_TRUE(solver.converged());
    // gap region still tracks the truth reasonably
    EXPECT_LT(rmse(solver.trajectory().middleRows(10, 10), fx.Ytrue.middleRows(10, 10)), 0.1);
}

// every admissible trajectory degree produces a valid, converged fit. The degree IS the discretization
// choice -- it selects the p-stage Gauss collocation scheme -- so this sweeps GL1 through GL4.
TEST(bs_ls_ode, trajectory_degrees_converge) {
    fixture fx;
    for (int degree = 1; degree <= 4; ++degree) {
        auto solver = fx.make_solver(degree);
        solver.fit(1.0);
        EXPECT_TRUE(solver.trajectory().allFinite()) << "degree " << degree;
        EXPECT_TRUE(solver.converged()) << "degree " << degree;
    }
}

/* The trajectory space depends on the mesh and the degree alone. Missing data changes nothing: every node
is a step node, so the fit is C0 at every node whether it is observed or not. The coefficients f() are d per
dof of that space. */
TEST(bs_ls_ode, trajectory_space_is_fixed_by_mesh_and_degree) {
    const int N = 40;   // the fixture's 41 nodes
    for (int degree = 1; degree <= 4; ++degree) {
        fixture fx;
        for (int t = 5; t < 15; t += 2) { fx.Yobs.row(t).setConstant(std::numeric_limits<double>::quiet_NaN()); }
        auto solver = fx.make_solver(degree);
        solver.fit(1.0);
        EXPECT_EQ(solver.degree(), degree);
        EXPECT_EQ(solver.n_dofs(), 2 * (degree * N + 1)) << "degree " << degree;
        EXPECT_EQ(solver.f().size(), solver.n_dofs()) << "degree " << degree;
    }
}

/* eval() is the basis expansion of f(), and that expansion IS the fit: at every node the coefficient of the
node's dof is the nodal state, and between nodes the expansion reproduces the step's collocation polynomial,
rebuilt here as the plain Lagrange interpolant of the fitted node and the step's stage values. */
TEST(bs_ls_ode, eval_is_the_basis_expansion_of_the_collocation_polynomial) {
    fixture fx;
    auto check = [&](int degree, auto tab) {
        auto solver = fx.make_solver(degree);
        solver.fit(1e-1);
        ASSERT_TRUE(solver.converged()) << "degree " << degree;
        const matrix_t& Y = solver.trajectory();
        const matrix_t& U = solver.control();
        const vector_t& c = solver.f();
        const BsSpace<Triangulation<1, 1>> Vh(fx.mesh, degree, std::vector<int>(fx.time.size(), degree));   // as fitted
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        controlled_ode_solver<S, Dynamic, nonlinear_field> engine(ode_rhs_field {nonlinear_field {}}, RKIntegrator<S>(tab));
        double err_nodes = 0, err_between = 0;
        for (int k = 0; k + 1 < Y.rows(); ++k) {
            const vector_t yk = Y.row(k).transpose(), uk = U.row(k).transpose();
            const int dof = Vh.dof_handler().active_dofs(k).front();
            err_nodes = std::max(err_nodes, (c.segment(dof * 2, 2) - yk).cwiseAbs().maxCoeff());
            // the step polynomial through y_n and the stage values, on the points {0, c_1, ..., c_s} of the
            // interval: a plain Lagrange interpolant, with no spline basis involved
            const double t0 = fx.time[k], h = fx.time[k + 1] - t0;
            const matrix_t Ys = engine.step_with_stage_values(t0, yk, h, matrix_t(Eigen::Map<const matrix_t>(uk.data(), 2, S))).values;
            std::vector<double> pts {0.0};
            std::vector<vector_t> vals {yk};
            for (int i = 0; i < S; ++i) {
                pts.push_back(tab.c()[i]);
                vals.push_back(Ys.col(i));
            }
            auto interpolant = [&](double theta) {
                vector_t v = vector_t::Zero(2);
                for (int a = 0; a <= S; ++a) {
                    double l = 1.0;
                    for (int b = 0; b <= S; ++b) {
                        if (b != a) { l *= (theta - pts[b]) / (pts[a] - pts[b]); }
                    }
                    v += l * vals[a];
                }
                return v;
            };
            // theta = 1 lands on the next node, where the expansion must agree with the interpolant's endpoint
            for (double theta : {0.13, 0.5, 0.91, 1.0}) {
                err_between =
                  std::max(err_between, (solver.eval(t0 + theta * h) - interpolant(theta)).cwiseAbs().maxCoeff());
            }
        }
        EXPECT_EQ(err_nodes, 0.0) << "degree " << degree;
        EXPECT_LT(err_between, 1e-11) << "degree " << degree;
    };
    check(1, ode_schemes::implicit_midpoint());
    check(2, ode_schemes::gauss_legendre_2());
    check(3, ode_schemes::gauss_legendre_3());
    check(4, ode_schemes::gauss_legendre_4());
}

/* Binding the response BY NAME. A system names its components, so each one reads its own column, and an
unknown the frame does not mention is simply unobserved -- which is what the auxiliary variable of a
higher-order system is, and what would otherwise force the caller to hand over a column of NaNs. */
TEST(bs_ls_ode, ode_system_binds_the_response_by_name) {
    fixture fx;
    const int m = static_cast<int>(fx.time.size());
    BsSpace<Triangulation<1, 1>> Vh(fx.mesh, 2, std::vector<int>(m, 2));
    ode_unknown x(Vh, "x"), v(Vh, "v");
    ode_system sys {dx(x) == v, dx(v) == -x};
    bs_ls_ode penalty(sys);   // the space comes from the unknowns
    // a system is not also handed a space: the two could disagree
    static_assert(!std::is_constructible_v<bs_ls_ode, decltype(sys), decltype(Vh)>);
    const vector_t xcol = fx.Yobs.col(0), vcol = fx.Yobs.col(1);

    {   // both components observed: one named column each
        GeoFrame data(fx.mesh);
        auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
        layer.load_vec("x", xcol);
        layer.load_vec("v", vcol);
        internals::bs_ls_ode solver;
        solver.discretize(penalty.get());
        solver.analyze_data("x ~ f", data);
        EXPECT_EQ(solver.n_components(), 2);   // the state dimension comes from the system, not the frame
        EXPECT_EQ(solver.n_obs(), 2 * m);
    }
    {   // only x observed: v is unobserved, and the caller supplied no NaNs to say so
        GeoFrame data(fx.mesh);
        auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
        layer.load_vec("x", xcol);
        internals::bs_ls_ode solver;
        solver.discretize(penalty.get());
        solver.analyze_data("x ~ f", data);
        EXPECT_EQ(solver.n_components(), 2);
        EXPECT_EQ(solver.n_obs(), m) << "only the named column is observed";
        solver.fit(1.0);
        EXPECT_TRUE(solver.trajectory().allFinite());
    }
}

/* End to end: a parameterized system drives the inverse solvers. This is what the leaf-level parameter
test cannot show -- that a system satisfies the is_parameterized_ode_rhs constraint bs_ls_ode_param
requires, that its declared parameters line up with the theta the descriptor binds, and that single
shooting recovers them from noiseless data generated by the system itself. The tolerance is loose because
a system carries no analytic param_jacobian: ode_rhs_field differentiates theta by central differences. */
TEST(bs_ls_ode, ode_system_drives_parameter_estimation) {
    fixture fx;   // for its mesh and time grid only; the data below comes from the system
    BsSpace<Triangulation<1, 1>> Vh(fx.mesh, 2, std::vector<int>(fx.time.size(), 2));
    ode_unknown p(Vh, "p"), q(Vh, "q");
    ode_parameter a0(1.0, "a0"), a1(1.0, "a1"), a2(1.0, "a2");
    ode_time t;
    ode_system sys {dx(p) == a0 * p * q + sin(t), dx(q) == a1 * p - a2 * q * q};
    sys.with_parameters(a0, a1, a2);
    ASSERT_EQ(sys.n_params(), 3);

    vector_t theta_true(3);
    theta_true << 1.0, 1.0, 1.0;
    vector_t y0(2);
    y0 << 0.5, -0.3;
    ode_rhs_field truth {sys, theta_true};   // the system at the true parameters
    const matrix_t Ytrue = RKIntegrator(ode_schemes::gauss_legendre_2()).integrate(truth, fx.time, y0);

    vector_t theta0(3);
    theta0 << 1.3, 0.7, 1.2;   // a perturbed start
    bs_ls_ode_param descriptor(sys, theta0, y0, /*max_iter=*/200, /*tol=*/1e-10);
    internals::bs_ls_ode_nls solver;
    solver.discretize(descriptor.get());
    solver.analyze_data(fx.time, Ytrue);
    EXPECT_EQ(solver.n_params(), 3);
    const vector_t theta_hat = solver.solve(theta0);
    EXPECT_LT((theta_hat - theta_true).cwiseAbs().maxCoeff(), 5e-2) << "theta_hat = " << theta_hat.transpose();
}

/* Parameters are what makes a system usable by the inverse solvers: they are declared in the order they
occupy theta, they read theta once one is bound, and they fall back to their declared values otherwise --
which is exactly the plain forward fit of a system an inverse solver would estimate. */
TEST(bs_ls_ode, ode_system_parameters_map_onto_theta) {
    fixture fx;
    BsSpace<Triangulation<1, 1>> Vh(fx.mesh, 2, std::vector<int>(fx.time.size(), 2));
    ode_unknown x(Vh, "x"), v(Vh, "v");
    ode_parameter c(0.5, "c"), k(2.0, "k");
    ode_system sys {dx(x) == v, dx(v) == -c * v - k * x};   // a damped linear oscillator
    sys.with_parameters(c, k);
    EXPECT_EQ(sys.n_params(), 2);
    EXPECT_EQ(c.index(), 0);   // declaration order IS the layout of theta
    EXPECT_EQ(k.index(), 1);

    vector_t y(2);
    y << 0.3, -0.4;
    vector_t theta(2);
    theta << 1.5, 3.0;
    const vector_t got = sys(0.0, y, theta);
    EXPECT_NEAR(got[0], y[1], 1e-15);                              // dx(x) == v
    EXPECT_NEAR(got[1], -1.5 * y[1] - 3.0 * y[0], 1e-15);          // theta, not the declared values
    const vector_t no_theta;
    const vector_t got0 = sys(0.0, y, no_theta);
    EXPECT_NEAR(got0[1], -0.5 * y[1] - 2.0 * y[0], 1e-15);         // unbound: the declared values
}

/* A system written in strong form is NOTATION, not a different model: ode_system compiles to the same
dynamics as the equivalent hand-written functor. The DYNAMICS are checked to round-off; the FITS only to
solver tolerance, and that difference is the point of the second comment below. The functor deliberately
carries no analytic Jacobian, since the system has none either -- both then differentiate by the same
central differences. */
TEST(bs_ls_ode, ode_system_matches_the_equivalent_functor) {
    struct plain_field {
        vector_t operator()(double t, const vector_t& y) const {
            vector_t out(2);
            out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
            return out;
        }
    };
    fixture fx;
    BsSpace<Triangulation<1, 1>> Vh(fx.mesh, 2, std::vector<int>(fx.time.size(), 2));
    ode_unknown a(Vh, "a"), b(Vh, "b");
    ode_time t;
    ode_system sys {dx(a) == a * b + sin(t), dx(b) == a - b * b};
    // the equation count is the system dimension, so the engine keeps its fixed-size stage math
    EXPECT_EQ(ode_rhs_dim_v<decltype(sys)>, 2);
    EXPECT_EQ(a.index(), 0);   // the state layout is the order of the left-hand sides
    EXPECT_EQ(b.index(), 1);

    // the dynamics themselves, at arbitrary states: this is what "the same model" means, and it holds to
    // round-off -- the expression tree evaluates the very same arithmetic the functor spells out
    plain_field field;
    const vector_t no_theta;   // this system declares no parameters
    double dyn_err = 0;
    for (double tt : {0.0, 0.37, 1.4, 2.0}) {
        for (double y0 : {-0.6, 0.2, 0.9}) {
            vector_t y(2);
            y << y0, 0.5 - y0;
            dyn_err = std::max(dyn_err, (sys(tt, y, no_theta) - field(tt, y)).cwiseAbs().maxCoeff());
        }
    }
    EXPECT_LT(dyn_err, 1e-15) << "the system must express the same dynamics as the functor";

    internals::bs_ls_ode from_system;
    internals::bs_ls_ode from_functor;
    {
        bs_ls_ode penalty(sys);
        from_system.discretize(penalty.get());
    }
    {
        bs_ls_ode penalty(plain_field {}, Vh);
        from_functor.discretize(penalty.get());
    }
    from_system.analyze_data(fx.time, fx.Yobs);
    from_functor.analyze_data(fx.time, fx.Yobs);
    from_system.fit(1.0);
    from_functor.fit(1.0);
    ASSERT_TRUE(from_system.converged());
    /* The FITS agree to solver tolerance, not bit for bit: the system knows dim = 2, so its engine runs the
    fixed-size stage math, while a VectorXd-returning functor runs the dynamic one. The two Eigen code paths
    round differently, and BFGS may stop anywhere within its tolerance. */
    EXPECT_LT((from_system.trajectory() - from_functor.trajectory()).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((from_system.f() - from_functor.f()).cwiseAbs().maxCoeff(), 1e-6);
}

/* The solver reads the space in discretize and keeps nothing of it, as fe_ls_elliptic does with its trial
space: only the mesh must outlive it. A solver whose space is gone fits, expands and evaluates exactly as one
whose space is alive. */
TEST(bs_ls_ode, the_space_is_read_at_discretize) {
    fixture fx;
    const int m = fx.time.size(), N = m - 1, p = 3;
    internals::bs_ls_ode solver;
    {
        BsSpace<Triangulation<1, 1>> Vh(fx.mesh, p, std::vector<int>(m, p));
        bs_ls_ode penalty(fx.f, Vh);
        solver.discretize(penalty.get());
    }   // space and descriptor are destroyed here
    solver.analyze_data(fx.time, fx.Yobs);
    solver.fit(1.0);
    ASSERT_TRUE(solver.converged());
    EXPECT_EQ(solver.n_dofs(), 2 * (p * N + 1));   // d per dof of the C0 space of degree p

    BsSpace<Triangulation<1, 1>> Vh(fx.mesh, p, std::vector<int>(m, p));
    bs_ls_ode penalty(fx.f, Vh);
    internals::bs_ls_ode reference;
    reference.discretize(penalty.get());
    reference.analyze_data(fx.time, fx.Yobs);
    reference.fit(1.0);

    const vector_t q = vector_t::LinSpaced(97, fx.time[0], fx.time[m - 1]);
    EXPECT_EQ((solver.f() - reference.f()).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((solver.eval(q) - reference.eval(q)).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((solver.control_coefficients() - reference.control_coefficients()).cwiseAbs().maxCoeff(), 0.0);
}

/* The fit is expanded in the trajectory space, so a space that does not contain it is an error: one left
smoother than C0 at any interior node is rejected as soon as it is discretized. */
TEST(bs_ls_ode, rejects_a_space_that_is_not_c0_at_every_node) {
    fixture fx;
    const int m = fx.time.size(), p = 2;
    std::vector<int> mu(m, p);
    mu[m / 2] = 1;   // a single interior node left smooth
    BsSpace<Triangulation<1, 1>> Vh(fx.mesh, p, mu);
    bs_ls_ode penalty(fx.f, Vh);
    internals::bs_ls_ode solver;
    EXPECT_DEATH(solver.discretize(penalty.get()), "Assertion");
}


// the NPRODE model wrapper drives the solver and selects lambda by GCV over a grid.
TEST(bs_ls_ode, model_wrapper_and_gcv) {
    fixture fx;
    using Model = NPRODE<internals::bs_ls_ode>;
    Triangulation<1, 1> T(fx.time);
    BsSpace Vh(T, 2, std::vector<int>(T.n_nodes(), 2));   // C0 at every node
    bs_ls_ode penalty(fx.f, Vh);
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

// GCV picks the *right* lambda: the GCV-optimal lambda tracks the oracle lambda -- the one minimizing the
// true error against the noise-free trajectory Ytrue (unavailable in practice, known here). The data are
// built so the bias/variance trade-off has an *interior* optimum: the truth deviates smoothly from any prior
// ODE solution (so very large lambda over-regularizes and biases) and is corrupted by genuine iid Gaussian
// noise (so very small lambda over-fits). Over a shared half-decade log grid the GCV choice lands within one
// grid step of the oracle and attains a true error within a small factor of the oracle minimum. End-to-end
// correctness check of the edf / GCV machinery (higher-order GL2 scheme -> exercises the forced-control edf).
TEST(bs_ls_ode, gcv_selects_near_oracle_lambda) {
    const int m = 41; const double T = 2.0;
    vector_t time = make_time(m, T);
    nonlinear_field f;
    vector_t y0(2); y0 << 0.5, -0.3;
    // off-manifold truth: a prior ODE solution plus a slow deviation the dynamics cannot represent
    matrix_t Ytrue = integrate_field(f, time, y0);
    for (int t = 0; t < m; ++t) {
        const double s = time[t];
        Ytrue(t, 0) += 0.12 * std::sin(1.5 * s);
        Ytrue(t, 1) += 0.08 * std::cos(1.1 * s);
    }
    // observations: truth + iid Gaussian noise (seeded, reproducible)
    std::mt19937 rng(2024u);
    std::normal_distribution<double> gauss(0.0, 0.03);
    matrix_t Yobs = Ytrue;
    for (int t = 0; t < m; ++t) {
        for (int v = 0; v < 2; ++v) { Yobs(t, v) += gauss(rng); }
    }
    const double raw_rmse = rmse(Yobs, Ytrue);

    using Model = NPRODE<internals::bs_ls_ode>;
    // the mesh outlives every model built below: their evaluation handles point at it
    Triangulation<1, 1> gcv_mesh(time);
    BsSpace gcv_space(gcv_mesh, 2, std::vector<int>(gcv_mesh.n_nodes(), 2));   // C0 at every node
    auto build = [&](Model& model) {
        bs_ls_ode penalty(f, gcv_space);
        model.discretize(penalty.get());
        model.analyze_data(time, Yobs);
    };
    // shared half-decade log grid over the numerically stable range: 10^{-2} ... 10^{3}
    std::vector<double> grid;
    for (int e = -4; e <= 6; ++e) { grid.push_back(std::pow(10.0, 0.5 * e)); }

    // oracle: the grid lambda minimizing RMSE against the truth
    Model oracle_model; build(oracle_model);
    int oracle_idx = 0; double best_rmse = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(grid.size()); ++i) {
        oracle_model.fit(grid[i]);
        double e = rmse(oracle_model.trajectory(), Ytrue);
        if (e < best_rmse) { best_rmse = e; oracle_idx = i; }
    }

    // GCV pick over the same grid
    Model model; build(model);
    auto gcv = model.gcv(200, 42);
    GridSearch<1> optimizer;
    optimizer.optimize(gcv, grid);
    double lambda_gcv = optimizer.optimum()[0];

    // locate the GCV pick on the grid (log-nearest)
    int gcv_idx = 0; double dmin = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(grid.size()); ++i) {
        double dd = std::abs(std::log10(grid[i]) - std::log10(lambda_gcv));
        if (dd < dmin) { dmin = dd; gcv_idx = i; }
    }
    model.fit(lambda_gcv);
    double gcv_rmse = rmse(model.trajectory(), Ytrue);

    std::cout << "[gcv] oracle_idx=" << oracle_idx << " (lambda=" << grid[oracle_idx]
              << ", rmse=" << best_rmse << ")  gcv_idx=" << gcv_idx << " (lambda=" << lambda_gcv
              << ", rmse=" << gcv_rmse << ")  raw_rmse=" << raw_rmse << std::endl;

    EXPECT_GT(oracle_idx, 0);                                  // interior optimum (not the weakest lambda)
    EXPECT_LT(oracle_idx, static_cast<int>(grid.size()) - 1);  // ... nor the strongest
    EXPECT_LE(std::abs(gcv_idx - oracle_idx), 1);              // GCV within one grid step of the oracle
    EXPECT_LT(gcv_rmse, 1.30 * best_rmse);                     // near-oracle true error
    EXPECT_LT(gcv_rmse, raw_rmse);                             // and it denoises
}

// end-to-end through an order-1 (time) GeoFrame and a formula: the multi-column response is
// read as a block column and the fit matches the plain-input path exactly.
TEST(bs_ls_ode, geoframe_formula_path) {
    fixture fx;
    const int m = fx.time.size();
    // order-1 time mesh matching fx.time, with the d-dimensional response as a block column
    Triangulation<1, 1> Tg = Triangulation<1, 1>::Interval(fx.time[0], fx.time[m - 1], m);
    GeoFrame data(Tg);
    auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", fx.Yobs);

    Triangulation<1, 1> T(fx.time);
    BsSpace Vh(T, 2, std::vector<int>(T.n_nodes(), 2));   // C0 at every node
    bs_ls_ode penalty(fx.f, Vh);
    NPRODE<internals::bs_ls_ode> model("y ~ f", data, penalty);
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
TEST(bs_ls_ode, edf_in_range_and_monotone) {
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
// core correctness check behind RKIntegrator::adjoint_step (hence the reduced-problem gradient). The
// additive control is passed to the (control-free) core as a parameter with identity Jacobian d f/d u = I,
// so the parameter gradient dC/dtheta returned under that map is exactly dC/du.
TEST(bs_ls_ode, adjoint_step_matches_finite_differences) {
    nonlinear_field f;
    const int d = 2;
    vector_t y(d), u(d), w(d);
    y << 0.4, -0.2;
    u << 0.1, -0.05;
    w << 1.3, -0.7;   // linear cost C = w . y_next, so the incoming costate is p_next = w
    const double t = 0.3, dt = 0.05, fd = 1e-6;
    auto identity_df_du = [d](double, const vector_t&) -> matrix_t { return matrix_t::Identity(d, d); };
    // exercised per scheme through a generic lambda: the tableaux have distinct stage counts and so
    // cannot share a single heterogeneous loop.
    auto check = [&](auto tab) {
        RKIntegrator integ(tab);
        ode_rhs_field field(f);
        auto step = [&](const vector_t& yy, const vector_t& uu) {
            return integ.step(operator_plus_c(field, uu), t, yy, dt);
        };
        rk_adj_step_t adj = integ.adjoint_step(operator_plus_c(field, u), t, y, dt, w, identity_df_du);
        const vector_t& p_curr = adj.costate;
        const vector_t& grad_contrib = adj.param_grad;
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

// the same dynamics as nonlinear_field but WITHOUT an analytic state_jacobian: routes ode_rhs_field through the
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
    Eigen::Matrix2d state_jacobian(double, const vector_t& y) const {
        Eigen::Matrix2d J;
        J << y[1], y[0], 1.0, -2.0 * y[1];
        return J;
    }
};

// an ode_rhs_field built from a callable WITHOUT an analytic Jacobian falls back to the MatrixField
// finite-difference Jacobian; it must agree with the analytic Jacobian of the same dynamics. Also
// checks the value path and that field + c offsets the value while preserving the (FD) Jacobian.
TEST(bs_ls_ode, ode_rhs_field_fd_jacobian_matches_analytic) {
    nonlinear_field f;                 // provides an analytic state_jacobian
    ode_rhs_field fd_field{nonlinear_field_no_jac{}};   // no state_jacobian -> FD Jacobian backend
    vector_t y(2), c(2);
    y << 0.4, -0.2;
    c << 0.5, -0.25;
    const double t = 0.3;
    EXPECT_LT((fd_field.state_jacobian(t, y) - f.state_jacobian(t, y)).cwiseAbs().maxCoeff(), 1e-4);   // FD Jacobian
    EXPECT_LT((fd_field(t, y) - f(t, y)).cwiseAbs().maxCoeff(), 1e-12);              // value path
    ode_rhs_field shifted = operator_plus_c(fd_field, c);
    EXPECT_LT((shifted.state_jacobian(t, y) - f.state_jacobian(t, y)).cwiseAbs().maxCoeff(), 1e-4);    // Jacobian unchanged by +c
    EXPECT_LT((shifted(t, y) - (f(t, y) + c)).cwiseAbs().maxCoeff(), 1e-12);         // value offset by c
}

// a field with a fixed-size return makes the stage math static: Dim is deduced (no n_components()),
// ode_rhs_field<2> and a controlled_ode_solver<Stages, 2> are built inside the (non-templated) solver,
// and the fit (through the fixed-size RKIntegrator stage math) denoises just like the dynamic path.
TEST(bs_ls_ode, static_dim_deduced_and_fits) {
    nonlinear_field_static sf;
    ode_rhs_field field{sf};                                  // Dim deduced from the Vector2d return
    static_assert(decltype(field)::dim == 2, "static dimension must be deduced from the field");
    vector_t y(2), c(2);
    y << 0.4, -0.2;
    c << 0.5, -0.25;
    // field algebra and Jacobian still hold on the static field
    EXPECT_LT((operator_plus_c(field, c)(0.3, y) - (field(0.3, y) + c)).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((operator_plus_c(field, c).state_jacobian(0.3, y) - field.state_jacobian(0.3, y)).cwiseAbs().maxCoeff(), 1e-12);

    // end-to-end fit through the static-dimension stack
    fixture fx;
    Triangulation<1, 1> T(fx.time);
    BsSpace Vh(T, 2, std::vector<int>(T.n_nodes(), 2));   // C0 at every node
    bs_ls_ode penalty(sf, Vh);   // Dim = 2 deduced internally, not in the type
    static_assert(decltype(penalty)::solver_t::n_lambda == 1);   // solver type is dimension-agnostic now
    internals::bs_ls_ode solver;
    solver.discretize(penalty.get());
    solver.analyze_data(fx.time, fx.Yobs);
    solver.fit(1.0);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT(rmse(solver.trajectory(), fx.Ytrue), rmse(fx.Yobs, fx.Ytrue));
}

// eval(t) reads the collocation polynomial the solver actually integrated, so it must agree with the
// fitted trajectory exactly at the grid nodes -- it interpolates the fit, it does not re-approximate it.
TEST(bs_ls_ode, eval_reproduces_the_fit_at_the_grid_nodes) {
    fixture fx;
    auto solver = fx.make_solver();
    solver.fit(1.0);
    ASSERT_TRUE(solver.converged());
    const matrix_t& Y = solver.trajectory();
    for (int i = 0; i < Y.rows(); ++i) {
        EXPECT_LT((solver.eval(fx.time[i]) - Y.row(i).transpose()).cwiseAbs().maxCoeff(), 1e-10) << "node " << i;
    }
    // vectorized overload agrees with the scalar one, on unsorted off-grid times
    vector_t q(4);
    q << 1.234, 0.111, 1.9, 0.5;
    matrix_t E = solver.eval(q);
    for (int i = 0; i < q.size(); ++i) {
        EXPECT_LT((E.row(i).transpose() - solver.eval(q[i])).cwiseAbs().maxCoeff(), 1e-14);
    }
}

// between nodes eval(t) must follow the FORCED dynamics the fit solved -- y' = f(t, y) + u_k on interval k --
// and not merely interpolate the nodal values geometrically. Checked against a finely resolved integration
// of those dynamics from the fitted node, and contrasted with linear interpolation of the same nodes: the
// collocation polynomial is orders of magnitude closer, which is the whole point of the interface.
TEST(bs_ls_ode, eval_follows_the_forced_dynamics_between_nodes) {
    fixture fx;
    auto solver = fx.make_solver();
    solver.fit(1.0);
    ASSERT_TRUE(solver.converged());
    const matrix_t& Y = solver.trajectory();
    const matrix_t& U = solver.control();
    ode_rhs_field field {fx.f};
    RKIntegrator reference(ode_schemes::gauss_legendre_3(), 200, 1e-15);

    double err_dense = 0, err_linear = 0;
    for (int k : {0, 7, 19, 33}) {
        const double t0 = fx.time[k], dt = fx.time[k + 1] - fx.time[k];
        vector_t yk = Y.row(k).transpose(), uk = U.row(k).transpose();
        for (double theta : {0.2, 0.5, 0.8}) {
            // the forced dynamics of this interval, integrated finely from the fitted node
            vector_t exact = reference
                               .integrate(operator_plus_c(field, uk), vector_t::LinSpaced(801, t0, t0 + theta * dt), yk)
                               .row(800)
                               .transpose();
            err_dense = std::max(err_dense, (solver.eval(t0 + theta * dt) - exact).cwiseAbs().maxCoeff());
            vector_t lin = (1.0 - theta) * yk + theta * vector_t(Y.row(k + 1).transpose());
            err_linear = std::max(err_linear, (lin - exact).cwiseAbs().maxCoeff());
        }
    }
    // the bound is the collocation interpolation error, O(dt^(s+1)) = O(dt^3) for GL2, with an O(1)
    // constant -- NOT the O(dt^(2s)) the nodes enjoy. Measured: 1.6e-5 against dt^3 = 1.3e-4. Do not
    // tighten this past the order bound; between nodes the polynomial cannot do better.
    const double dt = 2.0 / 40.0;
    EXPECT_LT(err_dense, std::pow(dt, 3)) << "dense " << err_dense;
    // and it is far better than what a caller could do unaided by interpolating the nodes (measured ~120x)
    EXPECT_GT(err_linear, 50 * err_dense) << "dense " << err_dense << " vs linear " << err_linear;
}

// refining the fit grid drives the continuous extension toward the truth, so eval is usable off-grid and
// not just a cosmetic wrapper: the fitted curve converges to the underlying trajectory at times that are
// never grid nodes of any of the fits.
TEST(bs_ls_ode, eval_converges_off_grid_under_grid_refinement) {
    const double T = 2.0;
    vector_t y0(2);
    y0 << 0.5, -0.3;
    // query times deliberately off every grid used below
    vector_t q(5);
    q << 0.313, 0.717, 1.093, 1.451, 1.887;
    // integrate() marches a grid, so each query time needs its own solve from y0 over [0, q_i]
    matrix_t truth(q.size(), 2);
    for (int i = 0; i < q.size(); ++i) {
        vector_t grid = vector_t::LinSpaced(2001, 0.0, q[i]);
        truth.row(i) = RKIntegrator(ode_schemes::gauss_legendre_3(), 200, 1e-15)
                         .integrate(ode_rhs_field {nonlinear_field {}}, grid, y0)
                         .row(2000);
    }
    double prev = 0;
    for (int m : {21, 41, 81}) {
        fixture fx(m, T, /*noise=*/0.0);   // noiseless: isolates discretization from smoothing bias
        auto solver = fx.make_solver();
        solver.fit(1e4);                   // large lambda: the fit is driven onto an ODE solution
        ASSERT_TRUE(solver.converged()) << "m = " << m;
        const double err = (solver.eval(q) - truth).cwiseAbs().maxCoeff();
        if (prev > 0) { EXPECT_LT(err, prev) << "m = " << m; }
        prev = err;
    }
}

// the fitted control is stage-wise: (m-1) x (s*d), and on a multi-stage scheme it genuinely varies WITHIN
// an interval -- the widened space is actually used, not just allocated. A 1-stage scheme is the control
// group: there the polynomial is degree 0 and the stage values must be exactly constant.
TEST(bs_ls_ode, fitted_control_is_stage_wise) {
    fixture fx;
    auto stage_spread = [&](int degree, int s) {
        auto solver = fx.make_solver(degree);
        solver.fit(1e-2);
        EXPECT_TRUE(solver.converged());
        const matrix_t& U = solver.control();
        EXPECT_EQ(U.rows(), 40);
        EXPECT_EQ(U.cols(), s * 2) << "control must carry s blocks of d";
        double spread = 0, mag = U.cwiseAbs().maxCoeff();
        for (int t = 0; t < U.rows(); ++t) {
            for (int v = 0; v < 2; ++v) {
                double mean = 0;
                for (int i = 0; i < s; ++i) { mean += U(t, i * 2 + v) / s; }
                for (int i = 0; i < s; ++i) { spread = std::max(spread, std::abs(U(t, i * 2 + v) - mean)); }
            }
        }
        return std::make_pair(spread, mag);
    };
    auto [spread1, mag1] = stage_spread(1, 1);
    EXPECT_EQ(spread1, 0.0) << "a 1-stage control is a constant by construction";
    EXPECT_GT(mag1, 0.0);
    auto [spread2, mag2] = stage_spread(2, 2);
    EXPECT_GT(spread2, 1e-3 * mag2) << "the degree-1 control should not collapse to a constant";
}

// eval_control is now a genuine polynomial evaluation, not a staircase: it returns the stage value at each
// collocation node (where the Lagrange basis is a selector) and interpolates between them.
TEST(bs_ls_ode, eval_control_interpolates_the_stage_values) {
    fixture fx;
    auto solver = fx.make_solver(2);
    solver.fit(1e-2);
    ASSERT_TRUE(solver.converged());
    const matrix_t& U = solver.control();
    const auto tab = ode_schemes::gauss_legendre_2();
    for (int k : {0, 11, 27, 39}) {
        const double t0 = fx.time[k], dt = fx.time[k + 1] - fx.time[k];
        for (int i = 0; i < 2; ++i) {   // at the nodes the basis is a selector: u(c_i) == u_i exactly
            vector_t at_node = solver.eval_control(t0 + tab.c()[i] * dt);
            EXPECT_LT((at_node - U.block(k, i * 2, 1, 2).transpose()).cwiseAbs().maxCoeff(), 1e-12)
              << "interval " << k << " stage " << i;
        }
        // between the nodes it is the interpolant, so it lies strictly between the stage values where
        // those differ -- a staircase would return one of them
        vector_t mid = solver.eval_control(t0 + 0.5 * dt);
        for (int v = 0; v < 2; ++v) {
            const double a = U(k, v), b = U(k, 2 + v);
            if (std::abs(a - b) > 1e-9) {
                EXPECT_GT(mid[v], std::min(a, b) - 1e-12) << "interval " << k;
                EXPECT_LT(mid[v], std::max(a, b) + 1e-12) << "interval " << k;
                EXPECT_GT(std::abs(mid[v] - a), 1e-12) << "not a staircase, interval " << k;
            }
        }
    }
}

/* eval_control() is the basis expansion of the control in the broken spline space of degree p-1 on the mesh,
and that expansion IS the fitted control: on every interval, nodes included (right limit), it reproduces the
Lagrange polynomial of the stage values, rebuilt here from the tableau independently of the solver. */
TEST(bs_ls_ode, eval_control_is_the_broken_basis_expansion) {
    fixture fx;
    const int N = 40;   // the fixture's 41 nodes
    auto check = [&](int degree, auto tab) {
        auto solver = fx.make_solver(degree);
        solver.fit(1e-1);
        ASSERT_TRUE(solver.converged()) << "degree " << degree;
        // broken, so p dofs per interval, d coefficients per dof
        EXPECT_EQ(solver.control_coefficients().size(), 2 * degree * N) << "degree " << degree;
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        const matrix_t& U = solver.control();
        double err = 0;
        for (int k = 0; k < N; ++k) {
            const double t0 = fx.time[k], h = fx.time[k + 1] - t0;
            for (double theta : {0.0, 0.07, 0.5, 0.93}) {
                vector_t u = vector_t::Zero(2);
                for (int i = 0; i < S; ++i) {
                    double l = 1.0;   // the Lagrange basis on the tableau nodes, l_i(theta)
                    for (int j = 0; j < S; ++j) {
                        if (j != i) { l *= (theta - tab.c()[j]) / (tab.c()[i] - tab.c()[j]); }
                    }
                    u += l * U.block(k, 2 * i, 1, 2).transpose();
                }
                err = std::max(err, (solver.eval_control(t0 + theta * h) - u).cwiseAbs().maxCoeff());
            }
        }
        EXPECT_LT(err, 1e-10) << "degree " << degree;
    };
    check(1, ode_schemes::implicit_midpoint());
    check(2, ode_schemes::gauss_legendre_2());
    check(3, ode_schemes::gauss_legendre_3());
    check(4, ode_schemes::gauss_legendre_4());
}

// refining the scheme (GL2 -> GL3) changes the fit only marginally: both discretize the SAME continuous
// optimal-control problem, so once the control space is rich enough the answer stops moving. This is the
// check that the stage-wise formulation converges to a scheme-independent limit rather than to whatever
// the tableau happens to impose.
TEST(bs_ls_ode, fit_is_stable_across_gauss_schemes) {
    fixture fx;
    for (double lambda : {1e-2, 1e0}) {
        auto gl2 = fx.make_solver(2);
        auto gl3 = fx.make_solver(3);
        gl2.fit(lambda);
        gl3.fit(lambda);
        ASSERT_TRUE(gl2.converged() && gl3.converged()) << "lambda = " << lambda;
        EXPECT_NEAR(gl2.objective(), gl3.objective(), 1e-6 * std::abs(gl3.objective())) << "lambda = " << lambda;
        EXPECT_LT(rmse(gl2.trajectory(), gl3.trajectory()), 1e-5) << "lambda = " << lambda;
        // and both evaluate to the same curve off the grid, where the two schemes share no structure
        vector_t q(3);
        q << 0.313, 0.947, 1.771;
        EXPECT_LT((gl2.eval(q) - gl3.eval(q)).cwiseAbs().maxCoeff(), 1e-4) << "lambda = " << lambda;
    }
}

// The control Jacobian d y_{n+1}/d u of a STAGE-WISE control, checked entry by entry against finite
// differences. This is the primitive the whole reduced gradient is assembled from, so if it is right for
// every stage block the optimizer's gradient is right; it is also the object whose shape changed from
// d x d to d x (s*d), which is exactly the kind of change finite differences catch.
TEST(bs_ode_engine, stage_wise_control_jacobian_matches_finite_differences) {
    for_each_gauss_scheme([](const char* name, auto tab) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        const int d = 2, sd = S * d;
        fdapde::controlled_ode_solver<S, fdapde::Dynamic, nonlinear_field> engine(ode_rhs_field {nonlinear_field {}}, RKIntegrator<S>(tab));
        vector_t y(d), u(sd);
        y << 0.4, -0.2;
        for (int k = 0; k < sd; ++k) { u[k] = 0.1 * std::cos(2.0 * k) - 0.03 * k; }   // distinct per stage
        const double t = 0.3, dt = 0.05, fd = 1e-6;
        // the engine takes a step's control as d x s; perturbations are applied to its stage-major flattening
        auto stages = [&](const vector_t& v) { return matrix_t(Eigen::Map<const matrix_t>(v.data(), d, S)); };
        fdapde::rk_fwd_step_t s = engine.step_with_jacobians(t, y, dt, stages(u));
        ASSERT_EQ(s.param.rows(), d);
        ASSERT_EQ(s.param.cols(), sd) << "control Jacobian must span all stages " << name;
        for (int k = 0; k < sd; ++k) {
            vector_t up = u, um = u;
            up[k] += fd;
            um[k] -= fd;
            vector_t col = (engine.step(t, y, dt, stages(up)) - engine.step(t, y, dt, stages(um))) / (2 * fd);
            for (int i = 0; i < d; ++i) { EXPECT_NEAR(s.param(i, k), col[i], 1e-6) << name << " entry " << i << "," << k; }
        }
        // and the flow block is unaffected by the control's extra degrees of freedom
        for (int j = 0; j < d; ++j) {
            vector_t yp = y, ym = y;
            yp[j] += fd;
            ym[j] -= fd;
            vector_t col = (engine.step(t, yp, dt, stages(u)) - engine.step(t, ym, dt, stages(u))) / (2 * fd);
            for (int i = 0; i < d; ++i) { EXPECT_NEAR(s.flow(i, j), col[i], 1e-6) << name << " flow " << i << "," << j; }
        }
    });
}

// A stage-wise control whose stage values are all EQUAL is a constant control, and must reproduce the
// constant-forcing dynamics exactly.
TEST(bs_ode_engine, equal_stage_values_reproduce_a_constant_control) {
    for_each_gauss_scheme([](const char* name, auto tab) {
        constexpr int S = std::decay_t<decltype(tab)>::n_stages();
        const int d = 2;
        fdapde::controlled_ode_solver<S, fdapde::Dynamic, nonlinear_field> engine(ode_rhs_field {nonlinear_field {}}, RKIntegrator<S>(tab));
        vector_t y(d), c(d);
        y << 0.4, -0.2;
        c << 0.13, -0.07;
        matrix_t u_rep(d, S);
        for (int i = 0; i < S; ++i) { u_rep.col(i) = c; }
        const double t = 0.3, dt = 0.05;
        // the same step taken against an explicitly constant forcing field
        ode_rhs_field constant_forced {nonlinear_field {}};
        vector_t ref = RKIntegrator<S>(tab).step(operator_plus_c(constant_forced, c), t, y, dt);
        EXPECT_LT((engine.step(t, y, dt, u_rep) - ref).cwiseAbs().maxCoeff(), 1e-13) << name;
        // the quadrature weights sum to one, so the penalty of a constant control is unchanged too:
        // sum_i b_i ||c||^2 = ||c||^2
        EXPECT_NEAR(engine.quad_weights().sum(), 1.0, 1e-14) << name;
    });
}
