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

#ifndef __BS_LS_ODE_SOLVER_H__
#define __BS_LS_ODE_SOLVER_H__

#include "header_check.h"

namespace fdapde {

// Control-aware ODE solver: the prior dynamics forced by an additive stage-wise control (g = f + u), stepped by
// the prior solver's own integrator.
// controlled_ode_solver<Stages, Dim, F> bundles a prior field ode_rhs_field<Dim, F> with an
// RKIntegrator<Stages> (held as an ode_solver over the prior field) and exposes exactly the single-step
// operations an optimal-control solver needs, with the per-interval control entering as an additive forcing
// of the dynamics. The stage count Stages, the system dimension Dim and the rhs functor type F are all
// concrete here, so the wrapped integrator keeps its fixed-stage / fixed-dimension fast paths and inlines
// the rhs; the type-erased any_controlled_ode_solver then hides all three parameters so any consumer holding
// it is free of the Stages, Dim and F template parameters.
template <int Stages, int Dim, typename F>
struct controlled_ode_solver {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    controlled_ode_solver() = default;
    controlled_ode_solver(ode_rhs_field<Dim, F> field, RKIntegrator<Stages> integrator) :
        solver_(std::move(field), std::move(integrator)) { }

    // forced forward step: integrate field + U over [t, t + dt], U d x s (U = 0 recovers the prior dynamics)
    vector_t step(double t, const vector_t& y, double dt, const matrix_t& U) const {
        return solver_.integrator().step(forced_(U, static_cast<int>(y.size()), t, dt), t, y, dt);
    }
    // full-horizon forward solve of the PRIOR (unforced) dynamics on the grid `times` from y0: Y (m x d)
    matrix_t solve(const vector_t& times, const vector_t& y0) const { return solver_.solve(times, y0); }
    // forced discrete adjoint of one step: U is the stage-wise control on the interval (d x s), p_next
    // the incoming costate; returns {p_curr, dC/du}. The control enters the integrator's parameter machinery
    // through control_df_du_, so dC/du is the core adjoint's parameter gradient under that map.
    rk_adj_step_t adjoint_step(
      double t, const vector_t& y, double dt, const vector_t& p_next, const matrix_t& U) const {
        const int d = static_cast<int>(y.size());
        return solver_.integrator().adjoint_step(forced_(U, d, t, dt), t, y, dt, p_next, control_df_du_(d, t, dt));
    }
    // forced step that also returns its stage values: for a collocation method the step's polynomial at the
    // tableau nodes, which with the step's start determine the whole solution on the interval
    rk_stage_step_t step_with_stage_values(double t, const vector_t& y, double dt, const matrix_t& U) const {
        return solver_.integrator().step_with_stage_values(forced_(U, static_cast<int>(y.size()), t, dt), t, y, dt);
    }
    int n_stages() const { return Stages; }
    // the tableau's quadrature weights b_i. On Gauss nodes the Gram matrix of the Lagrange basis is exactly
    // diagonal, int_0^1 l_i l_j = b_i delta_ij (the quadrature is exact to degree 2s-1 and l_i l_j has degree
    // 2s-2), so b_i is also what weights each stage in the L2 norm of a stage-wise control.
    vector_t quad_weights() const {
        vector_t b(Stages);
        for (int i = 0; i < Stages; ++i) { b[i] = solver_.integrator().tableau().b()[i]; }
        return b;
    }
    // the tableau nodes c_i in [0, 1]: where in the step each stage lives. A consumer representing a
    // function on the interval by its stage values needs them to place its own basis.
    vector_t nodes() const {
        vector_t c(Stages);
        for (int i = 0; i < Stages; ++i) { c[i] = solver_.integrator().tableau().c()[i]; }
        return c;
    }
    // unforced step together with the flow Jacobian of the prior dynamics (no control); used by edf()
    rk_fwd_step_t step_with_flow_jacobian(double t, const vector_t& y, double dt) const {
        return solver_.step_with_flow_jacobian(t, y, dt);
    }
    // forced step together with its state (flow) and control Jacobians (control-aware; u = 0 recovers the
    // prior dynamics). The forward-mode primitive the full-space SQP solver assembles its KKT system from.
    rk_fwd_step_t step_with_jacobians(
      double t, const vector_t& y, double dt, const matrix_t& U) const {
        const int d = y.size();
        return solver_.integrator().step_with_state_param_jacobians(
          forced_(U, d, t, dt), t, y, dt, control_df_du_(d, t, dt), Stages * d);
    }
    // forced parameter sensitivity: d y_{n+1}/d theta on the control-forced dynamics f + u for a
    // theta-dependent prior field. param_jacobian(t, y) -> R^{d x n_theta} is the parameter Jacobian of the
    // *unforced* rhs (the control u carries no theta), while the stage system is built on the forced
    // dynamics. This is the theta-analogue of step_with_jacobians' control block.
    template <typename ParamJacobian>
    matrix_t step_param_jacobian(
      double t, const vector_t& y, double dt, const matrix_t& U, const ParamJacobian& param_jacobian,
      int n_theta) const {
        return solver_.integrator().step_param_jacobian(
          forced_(U, static_cast<int>(y.size()), t, dt), t, y, dt, param_jacobian, n_theta);
    }
    // forced step together with BOTH its flow Jacobian d y_{n+1}/d y and its theta-parameter Jacobian
    // d y_{n+1}/d theta, from a SINGLE stage solve, using the field's own parameter Jacobian. The theta-analogue
    // of step_with_jacobians (which uses the stage selector for d y/d u): the tracking envelope gradient
    // needs, per interval, the state-transition matrix for the costate propagation and the parameter
    // sensitivity for grad_theta -- one shared stage factorization yields both instead of a separate
    // param_jacobian and adjoint_step. The concrete (non-templated) counterpart of step_param_jacobian, so it
    // crosses the type-erasure boundary.
    rk_fwd_step_t step_with_flow_param_jacobians(double t, const vector_t& y, double dt, const matrix_t& U) const {
        if constexpr (ode_rhs_field<Dim, F>::is_parametric()) {
            return solver_.integrator().step_with_state_param_jacobians(
              forced_(U, static_cast<int>(y.size()), t, dt), t, y, dt, [this](double tt, const vector_t& yy) { return solver_.field().param_jacobian(tt, yy); },
              solver_.field().n_params());
        } else {
            fdapde_assert(false && "step_with_flow_param_jacobians called on a non-parametric controlled_ode_solver");
            return rk_fwd_step_t {};
        }
    }

    // Parametric operations (meaningful when the prior field is theta-parameterized; the type-erased
    // any_controlled_ode_solver carries them for every field, so they are guarded to compile and no-op /
    // assert on a non-parametric field, which never calls them).
    // rebind the current parameter vector theta of the prior dynamics in place
    void set_theta(const vector_t& theta) {
        if constexpr (ode_rhs_field<Dim, F>::is_parametric()) { solver_.field().set_theta(theta); }
    }
    // number of parameters theta (0 for a non-parametric field)
    int n_params() const { return solver_.field().n_params(); }
    // d y_{n+1}/d theta on the control-forced dynamics f + u, using the field's own parameter Jacobian. The
    // concrete (non-templated) counterpart of step_param_jacobian, so it can cross the type-erasure boundary.
    matrix_t param_jacobian(double t, const vector_t& y, double dt, const matrix_t& U) const {
        if constexpr (ode_rhs_field<Dim, F>::is_parametric()) {
            return step_param_jacobian(
              t, y, dt, U, [this](double tt, const vector_t& yy) { return solver_.field().param_jacobian(tt, yy); },
              solver_.field().n_params());
        } else {
            fdapde_assert(false && "param_jacobian called on a non-parametric controlled_ode_solver");
            return matrix_t {};
        }
    }

    // the prior (unforced) field; the mutable overload lets a parametric consumer rebind theta in place
    const ode_rhs_field<Dim, F>& field() const { return solver_.field(); }
    ode_rhs_field<Dim, F>& field() { return solver_.field(); }

   private:
    // index of the tableau node nearest the step coordinate theta = (t - t0)/dt. The integrator evaluates the rhs
    // and its parameter Jacobian only at the stage times t0 + c_i*dt, where this recovers i exactly: reading the
    // stage-wise control needs no basis, only distinct nodes, which the Gauss schemes have.
    static int nearest_stage_(const std::array<double, Stages>& c, double t0, double dt, double t) {
        const double theta = (t - t0) / dt;
        int i = 0;
        for (int j = 1; j < Stages; ++j) {
            if (std::abs(theta - c[j]) < std::abs(theta - c[i])) { i = j; }
        }
        return i;
    }
    /* The prior dynamics forced by a stage-wise control on one step [t0, t0 + dt]: f(t, y) + u_i, i being the
    stage the integrator is evaluating (see nearest_stage_). It REFERENCES the prior field, the control and the
    tableau nodes rather than copying them, so it is a temporary of the engine call that builds it and must not
    outlive it; and, read only at the stage times, it is not a control between the nodes. */
    struct forced_rhs {
        static constexpr int dim = Dim;             // read by ode_rhs_dim
        const ode_rhs_field<Dim, F>* prior;
        const matrix_t* U;                          // d x s: column i is the control at node c_i
        const std::array<double, Stages>* c;        // tableau nodes
        double t0, dt;
        vector_t operator()(double t, const vector_t& y) const {
            return (*prior)(t, y) + U->col(nearest_stage_(*c, t0, dt, t));
        }
        // the control does not depend on y: the forced state Jacobian is the prior's
        matrix_t state_jacobian(double t, const vector_t& y) const { return prior->state_jacobian(t, y); }
    };
    ode_rhs_field<Dim, forced_rhs> forced_(const matrix_t& U, int d, double t0, double dt) const {
        fdapde_assert(U.rows() == d && U.cols() == Stages);   // one d-column per stage, d the state dimension
        return ode_rhs_field<Dim, forced_rhs>(forced_rhs {
          std::addressof(solver_.field()), std::addressof(U), std::addressof(solver_.integrator().tableau().c()), t0,
          dt});
    }
    /* d(f + u)/d vec(U) at a stage time: the d x (s*d) block row with the identity in the block of the stage being
    evaluated and zero elsewhere. vec(U) is the column-major flattening of the d x s control, i.e. its stage-major
    blocks, so the control Jacobian d y_{n+1}/d u and the gradient dC/du that the integrator's parameter machinery
    yields are laid out stage-major, block i for the control at node c_i. */
    auto control_df_du_(int d, double t0, double dt) const {
        const std::array<double, Stages>* c = std::addressof(solver_.integrator().tableau().c());
        return [c, d, t0, dt](double t, const vector_t&) {
            matrix_t D = matrix_t::Zero(d, Stages * d);
            D.block(0, nearest_stage_(*c, t0, dt, t) * d, d, d).setIdentity();
            return D;
        };
    }

    ode_solver<Stages, Dim, F> solver_;   // the control-free solver over the prior (unforced) dynamics
};

// type-erased control-aware solver: the solver-facing interface with Stages, Dim and F erased. Beyond the
// control-aware step/adjoint/jacobian primitives it also carries the parametric operations (set_theta,
// n_params, param_jacobian) so a single erased engine serves both the forward optimal-control solver and the
// parameter-estimation solver; on a non-parametric field the parametric methods are simply never called.
struct IControlledOdeSolver {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    template <typename T> using fn_ptrs =
      fdapde::bindings<&T::step, &T::adjoint_step, &T::step_with_flow_jacobian, &T::step_with_jacobians,
                       &T::set_theta, &T::n_params, &T::param_jacobian, &T::solve,
                       &T::step_with_flow_param_jacobians, &T::step_with_stage_values, &T::n_stages,
                       &T::quad_weights, &T::nodes>;
    vector_t step(double t, const vector_t& y, double dt, const matrix_t& U) const {
        return fdapde::invoke<vector_t, 0>(*this, t, y, dt, U);
    }
    matrix_t solve(const vector_t& times, const vector_t& y0) const {
        return fdapde::invoke<matrix_t, 7>(*this, times, y0);
    }
    rk_adj_step_t adjoint_step(
      double t, const vector_t& y, double dt, const vector_t& p_next, const matrix_t& U) const {
        return fdapde::invoke<rk_adj_step_t, 1>(*this, t, y, dt, p_next, U);
    }
    rk_fwd_step_t step_with_flow_jacobian(double t, const vector_t& y, double dt) const {
        return fdapde::invoke<rk_fwd_step_t, 2>(*this, t, y, dt);
    }
    rk_fwd_step_t step_with_jacobians(double t, const vector_t& y, double dt, const matrix_t& U) const {
        return fdapde::invoke<rk_fwd_step_t, 3>(*this, t, y, dt, U);
    }
    void set_theta(const vector_t& theta) { fdapde::invoke<void, 4>(*this, theta); }
    int n_params() const { return fdapde::invoke<int, 5>(*this); }
    matrix_t param_jacobian(double t, const vector_t& y, double dt, const matrix_t& U) const {
        return fdapde::invoke<matrix_t, 6>(*this, t, y, dt, U);
    }
    rk_fwd_step_t step_with_flow_param_jacobians(double t, const vector_t& y, double dt, const matrix_t& U) const {
        return fdapde::invoke<rk_fwd_step_t, 8>(*this, t, y, dt, U);
    }
    rk_stage_step_t step_with_stage_values(double t, const vector_t& y, double dt, const matrix_t& U) const {
        return fdapde::invoke<rk_stage_step_t, 9>(*this, t, y, dt, U);
    }
    int n_stages() const { return fdapde::invoke<int, 10>(*this); }
    vector_t quad_weights() const { return fdapde::invoke<vector_t, 11>(*this); }
    vector_t nodes() const { return fdapde::invoke<vector_t, 12>(*this); }
};

using any_controlled_ode_solver = fdapde::erase<fdapde::heap_storage, IControlledOdeSolver>;

namespace internals {

// Least-squares solver for ODE-penalized smoothing.
//
// Over a time grid t_1 < ... < t_m and a d-dimensional response, it fits the nodal trajectory
// Y in R^{m x d} minimizing
//
//   J = sum_{t,v} m_{t,v} (y_{t,v} - y^obs_{t,v})^2 + lambda * sum_{t=1}^{m-1} dt_t || u_t ||^2,
//
// where the control u(t) is an additive forcing of the prior dynamics y' = f(t, y) + u(t). The
// trajectory Y is not optimized directly: it is recovered by forward Runge-Kutta integration of
// f + u from the initial state (the core RKIntegrator), so the control u is the only decision
// variable (the reduced / adjoint-method formulation).

// The ODE system dimension and the integration scheme's stage count are not template parameters of the
// solver: they are erased into the any_controlled_ode_solver the penalty descriptor hands over (so a static-Dim
// field still drives the fixed-size stage math inside, without leaking the parameter here).
class bs_ls_ode {
   protected:   // protected (not private): the inverse solvers derive from this and drive the inner fit
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
    using space_t = BsSpace<Triangulation<1, 1>>;

    template <typename Penalty> struct is_valid_penalty {
        static constexpr bool value = requires(Penalty penalty) {
            penalty.engine();
            penalty.max_iter();
            penalty.tol();
        };
    };
    template <typename Penalty> static constexpr bool is_valid_penalty_v = is_valid_penalty<Penalty>::value;

   public:
    static constexpr int n_lambda = 1;
    using solver_category = ls_solver;
    /* Optimization strategy (runtime)
        - adjoint:  reduced-space BFGS driven by the consistent discrete-adjoint gradient (state
                    eliminated by forward integration).
        - sqp:      full-space Gauss-Newton Sequential Quadratic Programming (state Y and control u kept
                    as unknowns tied by the RK dynamics as equality constraints; forward-mode Jacobians,
                    no adjoint sweep).
    */
    enum class fit_policy { adjoint, sqp };

    bs_ls_ode() noexcept = default;
    // construct from formula + geoframe (time mesh and response read from the frame)
    template <typename GeoFrame, typename Penalty, typename WeightMatrix>
        requires(is_valid_penalty_v<Penalty>)
    bs_ls_ode(const std::string& formula, const GeoFrame& gf, Penalty&& penalty, const WeightMatrix&) {
        discretize(penalty);
        analyze_data(formula, gf);
    }
    template <typename GeoFrame, typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    bs_ls_ode(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty);
        analyze_data(formula, gf);
    }

    /* Discretize the penalty: store the engine and the scheme parameters, and read the trajectory space.
    As fe_ls_elliptic does with its trial space, the space is read here and not kept: its dof count, its mesh
    nodes, the basis sampled at every cell's stage points and an evaluation handle holding its own copy are
    extracted once, and the same for the control space built over the same mesh. The caller's space may go
    out of scope afterwards; its mesh may not, since the copies point at it. */
    template <typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    void discretize(Penalty&& penalty) {
        engine_ = penalty.engine();
        s_ = engine_.n_stages();
        degree_ = penalty.degree();
        fdapde_assert(degree_ == s_);   // a degree-p trajectory is realised by the p-stage Gauss scheme
        max_iter_ = penalty.max_iter();
        tol_ = penalty.tol();
        // the system dimension d is taken from the response (set in analyze_data), not the field
        if constexpr (requires(Penalty p) { p.has_ic(); }) {
            if (penalty.has_ic()) {
                has_ic_ = true;
                y0_ = penalty.ic();
            }
        }
        // component names, when the rhs form supplied them: they bind the response by name (analyze_data)
        component_names_ = penalty.component_names();
        fdapde_assert(penalty.space() && "bs_ls_ode requires a trajectory space");
        const space_t& Vh = *penalty.space();
        check_trajectory_space_(Vh);
        const Triangulation<1, 1>& mesh = Vh.triangulation();
        mesh_nodes_ = mesh.nodes().col(0);
        // the control's space: broken (multiplicity p = order + 1), degree p-1, over the same mesh
        const space_t Wh(mesh, degree_ - 1, std::vector<int>(mesh.n_nodes(), degree_));
        n_dofs_ = Vh.n_dofs();
        n_ctrl_dofs_ = Wh.n_dofs();
        stage_basis_ = sample_at_stages_(Vh);
        ctrl_stage_basis_ = sample_at_stages_(Wh);
        // store handles for basis system evaluation at times
        point_eval_ = [space = Vh](const matrix_t& locs) -> sparse_matrix_t {
            return internals::point_basis_eval(space, locs);
        };
        ctrl_point_eval_ = [space = Wh, nodes = mesh_nodes_](const matrix_t& locs) -> sparse_matrix_t {
            return broken_basis_eval_(space, nodes, locs);
        };
        coeff_valid_ = false;
        return;
    }

    // ingest the time grid (m x 1) and a multi-column response (m x d); NaN entries are missing
    void analyze_data(const vector_t& time_nodes, const matrix_t& y_obs) {
        fdapde_assert(time_nodes.size() >= 2 && y_obs.rows() == time_nodes.size());
        time_ = time_nodes;
        m_ = time_.size();
        for (int t = 0; t < m_ - 1; ++t) { fdapde_assert(time_(t + 1) - time_(t) > 0); }
        set_response_(y_obs);
        return;
    }
    // ingest from a formula over an order-1 (time only) geoframe
    template <typename GeoFrame, typename WeightMatrix>
    void analyze_data(const std::string& formula, const GeoFrame& gf, const WeightMatrix&) {
        analyze_data(formula, gf);
    }
    template <typename GeoFrame> void analyze_data(const std::string& formula, const GeoFrame& gf) {
        fdapde_static_assert(GeoFrame::Order == 1, THIS_CLASS_IS_FOR_ORDER_ONE_GEOFRAMES_ONLY);
        fdapde_assert(gf.n_layers() == 1);
        // time mesh: the single geometric index is time
        const auto& time_index = geo_index_cast<0, POINT>(gf[0]);
        const auto& time_coords = time_index.coordinates();
        m_ = time_coords.rows();
        fdapde_assert(m_ >= 2 && time_coords.cols() == 1);
        time_ = time_coords.col(0);
        for (int t = 0; t < m_ - 1; ++t) { fdapde_assert(time_(t + 1) - time_(t) > 0); }
        // response: a single column (named by the formula lhs) whose block holds the d system
        // components, read as an m x d matrix (mirrors the multivariate fpca convention)
        Formula formula_(formula);
        /* Binding the response to the state. By NAME when the rhs form named its components and the frame
        carries any of them: each component reads its own column, and one the frame does not mention is left
        NaN, i.e. unobserved. By POSITION otherwise: one column whose block holds the d components. */
        bool any_named = false;
        for (const std::string& name : component_names_) { any_named = any_named || gf[0].contains(name); }
        matrix_t y_obs;
        if (any_named) {
            const int d = static_cast<int>(component_names_.size());
            y_obs = matrix_t::Constant(m_, d, std::numeric_limits<double>::quiet_NaN());
            for (int v = 0; v < d; ++v) {
                if (!gf[0].contains(component_names_[v])) { continue; }   // this component is unobserved
                const matrix_t col = gf[0].data().template col<double>(component_names_[v]).as_matrix();
                fdapde_assert(col.rows() == m_ && col.cols() == 1);
                y_obs.col(v) = col.col(0);
            }
        } else {
            y_obs = gf[0].data().template col<double>(formula_.lhs()).as_matrix();
        }
        fdapde_assert(y_obs.rows() == m_);
        set_response_(y_obs);
        return;
    }

    // fit at a given lambda with the selected optimization strategy (default: adjoint). Both policies fill
    // the same result members, so edf()/rss()/observers behave identically afterwards.
    const vector_t& fit(double lambda, fit_policy policy = fit_policy::adjoint) {
        return policy == fit_policy::sqp ? fit_sqp_(lambda) : fit_adjoint_(lambda);
    }
    template <typename LambdaT>
        requires(internals::is_vector_like_v<LambdaT>)
    const vector_t& fit(LambdaT&& lambda, fit_policy policy = fit_policy::adjoint) {
        fdapde_assert(lambda.size() == n_lambda);
        return fit(lambda[0], policy);
    }

    // Optional state-variable box constraints lb <= y(t) <= ub, enforced at every time node (per component;
    // set a component of lb/ub to -/+ infinity to leave that side free). [LIMITED TO SQP]
    void set_state_bounds(const vector_t& lb, const vector_t& ub) {
        fdapde_assert(lb.size() == ub.size());
        y_lb_ = lb; y_ub_ = ub; has_state_bounds_ = true;
    }
    void clear_state_bounds() { has_state_bounds_ = false; }
    bool has_state_bounds() const { return has_state_bounds_; }

    /* GCV computation
    To compute the GCV index we need to linearise the objective around the solution (Y_, U_) found using the fit
    method. Such a linearization is required to estimate the effective degrees of freedom (edf) through the trace of the hat matrix. In fact, the relationship between data and fitted values is not linear, hence a linearization is needed.
    */
    // hutchinson approximation of Tr[S] for the linearized hat matrix S = M H^{-1} M
    double edf(int r = 100, int seed = random_seed) {
        fdapde_assert(m_ > 0 && lambda_ > 0);
        sparse_matrix_t H;
        assemble_(H);   // Gauss-Newton Hessian at the current solution (Y_, U_)
        Eigen::SparseLU<sparse_matrix_t> solver;
        solver.compute(H);
        fdapde_assert(solver.info() == Eigen::Success);
        int seed_ = (seed == random_seed) ? std::random_device()() : seed;
        std::mt19937 rng(seed_);
        rademacher_distribution rademacher;
        const int N = m_ * d_;
        double trS = 0;
        for (int k = 0; k < r; ++k) {
            vector_t u = vector_t::Zero(N);
            for (int t = 0; t < m_; ++t) {
                for (int v = 0; v < d_; ++v) {
                    if (mask_(t, v) != 0.0) { u(t * d_ + v) = rademacher(rng); }
                }
            }
            vector_t x = solver.solve(u);   // H^{-1} M u  (u already lives on observed entries)
            // M x restricted to observed entries, then u^T (M x)
            for (int t = 0; t < m_; ++t) {
                for (int v = 0; v < d_; ++v) {
                    if (mask_(t, v) != 0.0) { trS += u(t * d_ + v) * x(t * d_ + v); }
                }
            }
        }
        return trS / r;
    }

    // residual sum of squares over the observed entries (data term of the objective)
    double rss() const {
        double s = 0;
        for (int t = 0; t < m_; ++t) {
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { s += std::pow(Y_(t, v) - y_obs_(t, v), 2); }
            }
        }
        return s;
    }

    // observers
    int n_dofs() const { return n_dofs_ * d_; }   // d coefficients per dof of the trajectory space
    int n_obs() const { return n_obs_; }   // number of observed scalar entries
    int n_components() const { return d_; }
    int n_nodes() const { return m_; }
    const vector_t& time_nodes() const { return time_; }
    // expansion coefficients of the fitted trajectory in the trajectory space, dof-major: block j holds the d
    // components of coefficient j (see build_coefficients_)
    const vector_t& f() const {
        build_coefficients_();
        return f_;
    }
    const vector_t& fn() const { return fn_; }          // fitted at the observation nodes (m*d), node-major
    vector_t fitted() const { return fn_; }
    const vector_t& misfit() const { return g_; }       // flattened ODE-residual misfit ((m-1)*d)
    const vector_t& response() const { return y_; }     // flattened response (m*d), NaN -> 0
    const matrix_t& trajectory() const { return Y_; }   // m x d
    // additive control u_t ((m-1) x d): the decision variable forcing the dynamics as
    // f + u_t. This is the true control -- inverse solvers rely on it, since the outer sensitivities
    // are evaluated on the correctly forced dynamics. Distinct from misfit(), the finite-difference
    // defect (y_{t+1} - step_f(y_t))/dt against the *unforced* dynamics: the two coincide only for
    // forward Euler (see compute_misfit_).
    const matrix_t& control() const { return U_; }
    // dual variables of the state box constraints (m x d, zero where inactive / unbounded)
    const matrix_t& bound_multipliers() const { return bound_mult_; }
    double objective() const { return objective_value_; }
    int n_iter() const { return n_iter_; }
    bool converged() const { return converged_; }

    /* Continuous evaluation of the fitted trajectory: the basis expansion of f() in the trajectory space.
    Each Gauss step carries a degree-p collocation polynomial and consecutive steps share only their
    endpoint value, so the fit is an element of the space and the expansion reproduces the polynomial the
    solver integrated, to round-off, at ARBITRARY times -- without refitting and with no interpolation
    assumption bolted on top.

    Accuracy is O(dt^(p+1)) between nodes against O(dt^(2p)) at them (collocation superconvergence), so an
    off-grid value is one order less accurate than the grid it interpolates. That is a property of the
    polynomial, not of this routine. */
    // eval at many times at once (n x d); the times need not be sorted or lie on the grid
    matrix_t eval(const vector_t& times) const {
        fdapde_assert(m_ >= 2 && Y_.rows() == m_);
        build_coefficients_();
        return expand_at_(point_eval_, f_, n_dofs_, times);
    }
    vector_t eval(double t) const { return eval(vector_t::Constant(1, t)).row(0).transpose(); }
    /* The fitted control at time t: the basis expansion of control_coefficients() in the broken spline space
    of degree p-1 on the mesh. On each interval the control is the degree-(p-1) polynomial through its stage
    values, discontinuous at the nodes, and the expansion reproduces it exactly. At an interior node the value
    is the right limit. */
    vector_t eval_control(double t) const {
        fdapde_assert(m_ >= 2 && U_.rows() == m_ - 1 && U_.cols() == sd_());
        build_coefficients_();
        return expand_at_(ctrl_point_eval_, uc_, n_ctrl_dofs_, vector_t::Constant(1, t)).row(0).transpose();
    }

    int degree() const { return degree_; }
    // expansion coefficients of the fitted control in the broken space of degree p-1, dof-major (d per dof)
    const vector_t& control_coefficients() const {
        build_coefficients_();
        return uc_;
    }

   protected:
    // Common utilities
    // grid time step
    double dt_(int t) const { return time_(t + 1) - time_(t); }
    // width of one interval's control block: s stage values of d components each, stage-major
    int sd_() const { return s_ * d_; }
    /* The control is STAGE-WISE: on each interval, s stage values of d components, i.e. the degree-(s-1)
    polynomial through them, discontinuous at the nodes. The decision vector holds one d-sized block per stage
    of every interval, interval-major. Its L2 penalty sum_i b_i ||u_i||^2 is exact: ||u||^2 has degree 2s-2,
    within the tableau's s-point Gauss rule. */
    int n_blocks_() const { return (m_ - 1) * s_; }   // decision blocks: one per stage of every interval
    int n_ctrl_() const { return n_blocks_() * d_; }  // total control unknowns
    // the s stage values of interval t (d x s, column i at node c_i), unscaled from the decision vector z
    matrix_t expand_(const vector_t& z, int t) const {
        matrix_t u(d_, s_);
        for (int i = 0; i < s_; ++i) { u.col(i) = z.segment((t * s_ + i) * d_, d_) / ctrl_scale_(t * s_ + i); }
        return u;
    }
    // interval t of a control schedule stored row-wise as (m - 1) x (s*d), stage-major, as the d x s matrix the
    // engine takes -- and back
    matrix_t stage_block_(const matrix_t& U, int t) const {
        matrix_t M(d_, s_);
        for (int i = 0; i < s_; ++i) { M.col(i) = U.block(t, i * d_, 1, d_).transpose(); }
        return M;
    }
    void set_stage_block_(matrix_t& U, int t, const matrix_t& M) const {
        for (int i = 0; i < s_; ++i) { U.block(t, i * d_, 1, d_) = M.col(i).transpose(); }
    }
    // accumulate interval t's control gradient Btp (length s*d) into the decision gradient, as expand_'s transpose
    void accumulate_(vector_t& g, int t, const vector_t& Btp) const {
        for (int i = 0; i < s_; ++i) {
            g.segment((t * s_ + i) * d_, d_) += Btp.segment(i * d_, d_) / ctrl_scale_(t * s_ + i);
        }
        return;
    }
    // the tableau quadrature weights b_i, which weight the stages in the control's L2 norm
    vector_t quad_weights_() const { return engine_.quad_weights(); }
    // index of the interval of `grid` containing t (the last interval at the right endpoint)
    static int locate_in_(const vector_t& grid, double t) {
        const int n = static_cast<int>(grid.size());
        const int k = static_cast<int>(std::upper_bound(grid.data(), grid.data() + n, t) - grid.data()) - 1;
        return k < 0 ? 0 : (k > n - 2 ? n - 2 : k);
    }
    int locate_(double t) const { return locate_in_(time_, t); }        // control interval
    // one space's basis restricted to one cell, sampled at the cell's stage points
    struct cell_basis_t {
        std::vector<int> dofs;   // dofs active on the cell
        matrix_t values;         // s x dofs.size(), [values]_{ij} = \psi_{dofs[j]}(t_k + c_i * h_k)
    };
    std::vector<cell_basis_t> sample_at_stages_(const space_t& space) const {
        const vector_t c = engine_.nodes();   // tableau nodes c_i: where each stage lives in its step
        const int n_cells = mesh_nodes_.size() - 1;
        std::vector<cell_basis_t> cells(n_cells);
        for (int k = 0; k < n_cells; ++k) {
            cells[k].dofs = space.dof_handler().active_dofs(k);
            const int n = cells[k].dofs.size();
            const double h = mesh_nodes_[k + 1] - mesh_nodes_[k];
            cells[k].values.resize(s_, n);
            for (int i = 0; i < s_; ++i) {
                for (int j = 0; j < n; ++j) {
                    cells[k].values(i, j) = space.eval_cell_value(cells[k].dofs[j], mesh_nodes_[k] + c[i] * h);
                }
            }
        }
        return cells;
    }
    /* Basis evaluation for the broken control space. internals::point_basis_eval locates a point on closed
    cells, so a point on an interior node may land in the left cell, whose functions vanish there. The C0
    trajectory space is indifferent to that; a broken space is not, so here a node belongs to the cell on its
    right (the last node to the last cell), and the expansion takes the right limit. */
    static sparse_matrix_t broken_basis_eval_(const space_t& space, const vector_t& nodes, const matrix_t& locs) {
        sparse_matrix_t psi(locs.rows(), space.n_dofs());
        std::vector<Triplet<double>> triplet_list;
        for (int i = 0; i < locs.rows(); ++i) {
            for (int dof : space.dof_handler().active_dofs(locate_in_(nodes, locs(i, 0)))) {
                triplet_list.emplace_back(i, dof, space.eval_cell_value(dof, locs(i, 0)));
            }
        }
        psi.setFromTriplets(triplet_list.begin(), triplet_list.end());
        psi.makeCompressed();
        return psi;
    }
    // the expansion of dof-major coefficients (d per dof) at the given times, n x d
    template <typename BasisEval>
    matrix_t expand_at_(const BasisEval& basis_eval, const vector_t& coeff, int n_dofs, const vector_t& times) const {
        const double a = mesh_nodes_[0], b = mesh_nodes_[mesh_nodes_.size() - 1];
        matrix_t locs(times.size(), 1);
        for (int i = 0; i < times.size(); ++i) {
            fdapde_assert(times[i] >= time_(0) - eval_tol_ && times[i] <= time_(m_ - 1) + eval_tol_);
            locs(i, 0) = std::min(std::max(times[i], a), b);   // clamp onto the mesh: the basis vanishes outside it
        }
        const sparse_matrix_t Psi = basis_eval(locs);
        return Psi * Eigen::Map<const matrix_t>(coeff.data(), d_, n_dofs).transpose();
    }
    /* Expansion coefficients of the fit in the trajectory space, built lazily on the first f() or eval() after
    a fit and invalidated by the next one.
    On the C0 space the coefficient of a node's dof is the state at that node, read off Y_. The p - 1
    interior coefficients of cell k come from the step's stage values: for a collocation method they are the
    step's polynomial at the tableau nodes, and the cell's p + 1 active functions span the degree-p
    polynomials there, so matching them is exact for any basis of the space. The p stage values overdetermine
    the p - 1 unknowns by one and are matched in least squares, which is exact when the nodes satisfy the
    dynamics; under the SQP policy they do so only to the constraint tolerance, and the expansion follows Y_
    at the nodes. */
    void build_coefficients_() const {
        if (coeff_valid_) { return; }
        fdapde_assert(Y_.rows() == m_ && U_.rows() == m_ - 1 && U_.cols() == sd_());
        fdapde_assert(static_cast<int>(stage_values_.size()) == m_ - 1 && "no fit has captured its dynamics");
        const int p = degree_;
        f_ = vector_t::Zero(n_dofs_ * d_);
        for (int k = 0; k < m_ - 1; ++k) {
            const cell_basis_t& cell = stage_basis_[k];
            fdapde_assert(static_cast<int>(cell.dofs.size()) == p + 1);
            const vector_t yk = Y_.row(k).transpose(), yk1 = Y_.row(k + 1).transpose();
            f_.segment(cell.dofs[0] * d_, d_) = yk;
            f_.segment(cell.dofs[p] * d_, d_) = yk1;
            if (p < 2) { continue; }
            const matrix_t& Ys = stage_values_[k];   // captured by the fit (see capture_dynamics_)
            // p x d right-hand side: the stage values less the contribution of the two node dofs
            const matrix_t R =
              Ys.transpose() - cell.values.col(0) * yk.transpose() - cell.values.col(p) * yk1.transpose();
            // (p - 1) x d interior coefficients, least squares
            const matrix_t C = cell.values.middleCols(1, p - 1).householderQr().solve(R);
            for (int j = 1; j < p; ++j) { f_.segment(cell.dofs[j] * d_, d_) = C.row(j - 1).transpose(); }
        }
        /* The control, in the broken space of degree p-1. Cell k carries p active functions of its own, which
        span the degree-(p-1) polynomials there, and the control polynomial takes its stage values at the
        tableau nodes, so collocating at those nodes is exact. */
        uc_ = vector_t::Zero(n_ctrl_dofs_ * d_);
        for (int k = 0; k < m_ - 1; ++k) {
            const cell_basis_t& cell = ctrl_stage_basis_[k];
            fdapde_assert(static_cast<int>(cell.dofs.size()) == p);
            matrix_t R(p, d_);
            for (int i = 0; i < p; ++i) { R.row(i) = U_.block(k, i * d_, 1, d_); }
            const matrix_t C = cell.values.partialPivLu().solve(R);   // p x d
            for (int j = 0; j < p; ++j) { uc_.segment(cell.dofs[j] * d_, d_) = C.row(j).transpose(); }
        }
        coeff_valid_ = true;
        return;
    }
    // slack on the time-range check in eval(), in units of the horizon: guards against a caller's endpoint
    // being a rounding step outside [t_0, t_m-1]
    static constexpr double eval_tol_ = 1e-12;
    /* Control rescaling (diagonal preconditioner for the reduced solve)
    BFGS seeds its inverse Hessian with the IDENTITY (see core BFGS::optimize) and takes a first step
    -h * grad, so the decision variable must be scaled to make the reduced Hessian O(1). Optimizing in
    v_t = s_t u_t achieves that when s_t^2 approximates the diagonal of d^2 J / d u_t^2.

    That Hessian has two parts: the penalty block, exactly 2 lambda dt_t b_i I for stage i, and the data block, which is
    O(1) in lambda. Scaling by the penalty block ALONE (s_t^2 = 2 lambda dt_t) is correct only while the
    penalty dominates: as lambda -> 0 the penalty block vanishes but the data block does not, so 1/s_t
    over-amplifies and the first step blows the forward integration up. Hence the data floor kappa_t --
    a cheap surrogate for the data block's diagonal. The sensitivity of a downstream node to u_t is
    dominated by B_t = d y_{t+1}/d u_t ~ dt_t I, and each observed scalar downstream contributes ~2, so

        kappa_t = 2 dt_t^2 (observed scalars at nodes > t) / d,   s_{t,i} = sqrt(2 lambda dt_t b_i + kappa_t)

    This is a preconditioner, so an order-of-magnitude surrogate is enough; it is negligible at large
    lambda (where the penalty-only scale is already right) and takes over as lambda -> 0.

    With a floor the penalty is no longer exactly ||v||^2/2, so the objective and gradient carry the
    weight w_{t,i} = 2 lambda dt_t b_i / s_{t,i}^2 in (0, 1]; w == 1 recovers the pure penalty scaling. The
    rescaling is internal to the reduced solve: the control is unscaled back to u_t = v_t / s_t before it
    reaches the engine, and U_ / control() stay physical, so every consumer (misfit, GCV, the inverse
    solvers) is unaffected. */
    double ctrl_scale_(int blk) const { return ctrl_s_[blk]; }
    double ctrl_weight_(int blk) const { return ctrl_w_[blk]; }
    // build the per-block scales s_{t,i} and penalty weights w_{t,i}; call once per fit, after lambda_ is set
    void build_ctrl_scale_() {
        ctrl_s_.resize(n_blocks_());
        ctrl_w_.resize(n_blocks_());
        // observed scalar entries strictly downstream of each interval
        std::vector<double> n_obs_after(m_, 0.0);
        for (int t = m_ - 2; t >= 0; --t) {
            double cnt = 0;
            for (int v = 0; v < d_; ++v) {
                if (mask_(t + 1, v) != 0.0) { cnt += 1.0; }
            }
            n_obs_after[t] = n_obs_after[t + 1] + cnt;
        }
        const vector_t b = quad_weights_();
        for (int t = 0; t < m_ - 1; ++t) {
            const double kappa = 2.0 * dt_(t) * dt_(t) * n_obs_after[t] / d_;
            for (int i = 0; i < s_; ++i) {
                // the penalty block of stage i is 2 lambda dt_t b_i I: the Gram matrix of the Lagrange basis at
                // Gauss nodes is exactly diag(b_i)
                const double pen = 2.0 * lambda_ * dt_(t) * b[i];
                ctrl_s_[t * s_ + i] = std::sqrt(pen + kappa);
                ctrl_w_[t * s_ + i] = pen / (pen + kappa);
            }
        }
        return;
    }
    // Fill mask and response variable from raw data
    void set_response_(const matrix_t& y_obs) {
        d_ = y_obs.cols();
        fdapde_assert(d_ > 0);
        y_obs_ = y_obs;
        mask_ = matrix_t::Ones(m_, d_);
        y_fill_ = y_obs;
        n_obs_ = 0;
        for (int t = 0; t < m_; ++t) {
            for (int v = 0; v < d_; ++v) {
                if (std::isnan(y_obs_(t, v))) {
                    mask_(t, v) = 0.0;
                    y_fill_(t, v) = 0.0;
                } else {
                    ++n_obs_;
                }
            }
        }
        // flattened response (missing -> 0)
        y_.resize(m_ * d_);
        for (int t = 0; t < m_; ++t) { y_.segment(t * d_, d_) = y_fill_.row(t).transpose(); }
        check_mesh_();
        // any capture belongs to the previous data set (see capture_dynamics_)
        stage_values_.clear();
        jac_flow_.clear();
        jac_B_.clear();
        coeff_valid_ = false;
        return;
    }
    /* Check the trajectory space: it must be the C0 spline space of degree p (its mesh is checked against
    the data by check_mesh_). The fit is expanded in this space, so the discrete trajectory must lie in it
    exactly. Each Gauss step carries a degree-p polynomial and consecutive steps share only their endpoint
    value, so that space has knot multiplicity p at EVERY interior node, observed or not. The continuous
    minimiser is smoother away from the data (its costate jumps only where data enters), but the discrete
    fit's extra kinks vanish under refinement and, like the gradient jumps of a P1 finite element solution,
    are not encoded in the space. Boundary entries are ignored: a clamped knot vector repeats the boundary
    nodes order + 1 times anyway. */
    void check_trajectory_space_(const space_t& Vh) const {
        const Triangulation<1, 1>& mesh = Vh.triangulation();
        fdapde_assert(mesh.n_nodes() >= 2 && Vh.order() == degree_);
        const std::vector<int>& mu = Vh.node_multiplicity();
        for (int t = 1; t + 1 < mesh.n_nodes(); ++t) {
            fdapde_assert(mu[t] == degree_ && "the trajectory space must be C0 at every interior node (mu = p)");
        }
        return;
    }
    // the data must live on the mesh the trajectory space was built over: the same nodes, to rounding
    void check_mesh_() const {
        fdapde_assert(mesh_nodes_.size() == m_ && "the data must live on the trajectory space's mesh");
        double dev = 0;
        for (int t = 0; t < m_; ++t) { dev = std::max(dev, std::abs(mesh_nodes_[t] - time_(t))); }
        fdapde_assert(dev <= 1e-10 * (time_(m_ - 1) - time_(0)));
        return;
    }
    // per-component linear interpolation/extrapolation of the observed values (start point)
    matrix_t initial_guess_() const {
        matrix_t Y(m_, d_);
        for (int v = 0; v < d_; ++v) {
            std::vector<int> obs;
            for (int t = 0; t < m_; ++t) {
                if (mask_(t, v) != 0.0) { obs.push_back(t); }
            }
            if (obs.empty()) {
                double fill = has_ic_ ? y0_(v) : 0.0;
                for (int t = 0; t < m_; ++t) { Y(t, v) = fill; }
                continue;
            }
            for (int t = 0; t < m_; ++t) {
                if (t <= obs.front()) {
                    Y(t, v) = y_obs_(obs.front(), v);
                } else if (t >= obs.back()) {
                    Y(t, v) = y_obs_(obs.back(), v);
                } else {
                    int lo = obs.front(), hi = obs.back();
                    for (int k : obs) {
                        if (k <= t) { lo = k; }
                        if (k >= t) { hi = k; break; }
                    }
                    double w = (hi == lo) ? 0.0 : (time_(t) - time_(lo)) / (time_(hi) - time_(lo));
                    Y(t, v) = (1 - w) * y_obs_(lo, v) + w * y_obs_(hi, v);
                }
            }
        }
        return Y;
    }

    // ODE-residual misfit ((m-1) x d): the finite-difference defect (y_{t+1} - step_f(y_t))/dt of the
    // fitted trajectory against the *unforced* prior dynamics f (zero forcing). A scheme-independent
    // diagnostic, exposed (flattened) as misfit(). It coincides with the additive control control()/U_
    // only for forward Euler -- a general RK scheme samples f at the forcing-shifted stage states, so
    // the two differ. Reported only; inverse solvers use control()/U_, the correctly forced control.
    void compute_misfit_() {
        misfit_.resize(m_ - 1, d_);
        for (int t = 0; t < m_ - 1; ++t) {
            double dt = dt_(t);
            vector_t yc = Y_.row(t).transpose();
            // prior dynamics only (no control): pass a zero forcing
            vector_t step = engine_.step(time_(t), yc, dt_(t), matrix_t::Zero(d_, s_));
            misfit_.row(t) = ((Y_.row(t + 1).transpose() - step) / dt).transpose();
        }
        return;
    }
    /* Capture, from the fitted (Y_, U_), the per-interval dynamics the post-fit consumers need: the stage
    values the trajectory's expansion is built from (build_coefficients_) and the forced step Jacobians the
    linearized hat matrix is built from (assemble_). One pass at the end of every fit, so that neither an
    observer nor the GCV path re-runs the dynamics. */
    void capture_dynamics_() {
        stage_values_.resize(m_ - 1);
        jac_flow_.resize(m_ - 1);
        jac_B_.resize(m_ - 1);
        for (int t = 0; t < m_ - 1; ++t) {
            const vector_t yc = Y_.row(t).transpose();
            const matrix_t ut = stage_block_(U_, t);
            stage_values_[t] = engine_.step_with_stage_values(time_(t), yc, dt_(t), ut).values;
            auto s = engine_.step_with_jacobians(time_(t), yc, dt_(t), ut);
            jac_flow_[t] = s.flow;
            jac_B_[t] = s.param;
        }
        return;
    }
    void flatten_() {
        fn_.resize(m_ * d_);
        for (int t = 0; t < m_; ++t) { fn_.segment(t * d_, d_) = Y_.row(t).transpose(); }
        g_.resize((m_ - 1) * d_);
        for (int t = 0; t < m_ - 1; ++t) { g_.segment(t * d_, d_) = misfit_.row(t).transpose(); }
        coeff_valid_ = false;   // the expansion of the previous fit (see build_coefficients_)
        return;
    }

    // Adjoint method
    // Decision vector layout: z = [v_{0,1}; ...; v_{0,s}; ...; v_{m-2,s}; (y_1)], one d-sized block per stage of
    // every interval, optionally followed by the free initial state y_1 (absent under a hard IC).
    // The control blocks hold the RESCALED control v_{t,i} = s_{t,i} u_{t,i}, s_{t,i} = ctrl_scale_(t * s + i);
    // the y_1 block carries no penalty and is left unscaled (its gradient is the costate at node 0,
    // already data-scaled).

    // Recover the state trajectory at z, with one-slot memoization. Besides the trajectory Y it caches, per
    // interval, the forward-mode Jacobians flow_t = d y_{t+1}/d y_t and B_t = d y_{t+1}/d u_t -- all from a
    // SINGLE stage solve per interval (step_with_jacobians reuses one stage factorization for the state and
    // the control sensitivity blocks). The BFGS/Wolfe optimizer evaluates the reduced objective and its
    // adjoint gradient at the *same* z (the line search queries obj(z) and grad(z) at each trial), so caching
    // on z collapses what were three stage passes per point -- objective's forward, gradient's forward, and
    // the gradient's backward adjoint stage re-solve -- into one. The gradient's backward sweep then needs no
    // stage solves at all: it contracts the stored flow_t / B_t against the costate (see gradient_). Not const:
    // it updates the cache (fwd_*), which is reset at the top of each fit_adjoint_ (a new lambda, theta or data
    // set invalidates the stored pass).
    const matrix_t& forward_recover_(const vector_t& z) {
        if (fwd_valid_ && fwd_z_.size() == z.size() && fwd_z_ == z) { return fwd_Y_; }
        const int nu = n_ctrl_();
        vector_t y0 = has_ic_ ? y0_ : vector_t(z.segment(nu, d_));
        fwd_Y_.resize(m_, d_);
        fwd_Y_.row(0) = y0.transpose();
        fwd_flow_.resize(m_ - 1);
        fwd_B_.resize(m_ - 1);
        for (int t = 0; t < m_ - 1; ++t) {
            vector_t yc = fwd_Y_.row(t).transpose();
            const matrix_t u_t = expand_(z, t);   // decision variables -> physical stage control (d x s)
            // forward step with its state (flow) and control (B) Jacobians, from one stage solve. On a
            // divergent control the state turns non-finite and NaN propagates down the trajectory; the
            // callers guard on fwd_Y_.allFinite() before touching the Jacobians.
            auto s = engine_.step_with_jacobians(time_(t), yc, dt_(t), u_t);
            fwd_Y_.row(t + 1) = s.state.transpose();
            fwd_flow_[t] = s.flow;
            fwd_B_[t] = s.param;
        }
        fwd_z_ = z;
        fwd_valid_ = true;
        return fwd_Y_;
    }

    // control objective J = SSE(observed) + lambda * sum_t dt_t ||u_t||^2
    // (state is integrated out)
    double objective_(const vector_t& z) {
        const matrix_t& Y = forward_recover_(z);
        // a control may drive the (possibly nonlinear) forward integration to blow up: report a
        // large finite cost so the line search backtracks out of the divergent region.
        if (!Y.allFinite()) { return std::numeric_limits<double>::infinity(); }
        double J = 0;
        for (int t = 0; t < m_; ++t) {
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { J += std::pow(Y(t, v) - y_obs_(t, v), 2); }
            }
        }
        // penalty lambda * sum_t dt_t sum_i b_i ||u_{t,i}||^2, written in the SCALED variables. Each decision
        // block is one stage of one interval, so the penalty is diagonal in z: 0.5 * sum_blk w_blk ||z_blk||^2,
        // with w_blk = 2 lambda dt b_i / s_blk^2.
        for (int blk = 0; blk < n_blocks_(); ++blk) {
            J += 0.5 * ctrl_weight_(blk) * z.segment(blk * d_, d_).squaredNorm();
        }
        return J;
    }

    // control gradient dJ/dz by a backward discrete-adjoint sweep, using the forward-mode Jacobians cached by
    // forward_recover_ -- no stage system is re-solved here. Node data-source s_t = 2 * mask_t * (y_t - y^obs_t);
    // the costate satisfies p_t = s_t + flow_t^T p_{t+1} with p_m = s_m, and the per-interval control gradient
    // is dJ_data/du_t = B_t^T p_{t+1} (the reverse-mode contraction of the forward control Jacobian).
    vector_t gradient_(const vector_t& z) {
        const int nu = n_ctrl_();
        const bool free_y1 = !has_ic_;
        const matrix_t& Y = forward_recover_(z);
        // divergent control: return a finite (zero) gradient so the optimizer state stays clean;
        // the matching large objective makes the line search reject the step.
        if (!Y.allFinite()) { return vector_t::Zero(nu + (free_y1 ? d_ : 0)); }
        auto source = [&](int t) {
            vector_t s = vector_t::Zero(d_);
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { s(v) = 2.0 * (Y(t, v) - y_obs_(t, v)); }
            }
            return s;
        };
        vector_t g = vector_t::Zero(nu + (free_y1 ? d_ : 0));
        for (int blk = 0; blk < n_blocks_(); ++blk) {   // diagonal penalty gradient, added up front
            g.segment(blk * d_, d_) = ctrl_weight_(blk) * z.segment(blk * d_, d_);
        }
        vector_t p = source(m_ - 1);
        for (int t = m_ - 2; t >= 0; --t) {
            // p is the costate p_{t+1}. In the rescaled control the chain rule gives
            //   dJ/dv_t = (1/s_t) dJ/du_t = (1/s_t) (2 lambda dt_t u_t + B_t^T p) = w_t v_t + B_t^T p / s_t,
            // with w_t = 2 lambda dt_t / s_t^2. Then propagate the costate by flow_t^T p.
            // data part: contract B_t^T p through the transpose of expand_
            // (the penalty part is diagonal and was added above)
            accumulate_(g, t, vector_t(fwd_B_[t].transpose() * p));
            p = source(t) + fwd_flow_[t].transpose() * p;
        }
        if (free_y1) { g.segment(nu, d_) = p; }   // dJ/dy_1 = costate at the first node
        return g;
    }
    // objective functor over the control z, adapting J and its gradient to the core BFGS interface. Holds a
    // non-const solver pointer: objective_ / gradient_ populate the forward-pass cache (forward_recover_).
    struct control_objective {
        bs_ls_ode* solver;
        double operator()(const vector_t& z) const { return solver->objective_(z); }
        auto gradient() const {
            return [s = solver](const vector_t& z) { return s->gradient_(z); };
        }
    };
    /* Best-iterate tracker, as a core optimizer callback.
    BFGS returns its LAST iterate, which is not necessarily its best (see the guard in fit_adjoint_), so
    the solver needs the best point the run actually visited. This records it through the optimizer's own
    callback interface -- grad_hook fires once per accepted step, after x_new and grad_new are in place --
    rather than by having objective_ mutate solver state as a side effect of being evaluated.

    Re-evaluating the objective here is nearly free: the forward pass is memoized on z, and grad_new was
    just computed at this very point, so obj(x_new) hits that cache and only re-sums misfit and penalty.

    Scope note: this sees the ACCEPTED iterates, not the line search's trial points, which the Wolfe
    callback evaluates internally. That is the principled candidate set -- a trial the line search rejected
    is not a point the optimizer ever stood on -- and it covers the failure this exists for, where the
    accepted sequence walks into a blow-up and the objective turns non-finite there. */
    struct best_iterate {
        bs_ls_ode* solver;
        template <typename Opt, typename Obj> bool grad_hook(Opt& opt, Obj& obj) {
            // pre-loop call: x_new is still the NaN placeholder, x0 is seeded by fit_adjoint_ instead
            if (!opt.x_new.allFinite()) { return false; }
            const double J = obj(opt.x_new);
            if (std::isfinite(J) && (!solver->has_best_ || J < solver->best_J_)) {
                solver->best_z_ = opt.x_new;
                solver->best_J_ = J;
                solver->has_best_ = true;
            }
            return false;   // never forces a stop
        }
    };

    // Optimal control formulation: the decision variable is the
    // stage-wise control u that drives the prior dynamics y' = f(t, y) + u(t); the
    // trajectory is recovered by forward integration and the gradient of the data misfit w.r.t.
    // u by the consistent discrete adjoint of the RK scheme.
    const vector_t& fit_adjoint_(double lambda) {
        fdapde_assert(lambda > 0 && d_ > 0 && m_ >= 2 && static_cast<bool>(engine_));
        if (has_ic_) { fdapde_assert(y0_.size() == d_); }
        lambda_ = lambda;
        build_ctrl_scale_();   // diagonal preconditioner for the reduced solve (depends on lambda)
        fwd_valid_ = false;   // stale across a new lambda / theta / data set
        has_best_ = false;    // best-finite-iterate tracker, per solve
        coeff_valid_ = false;   // the expansion of the previous fit (see build_coefficients_)
        const int sd = s_ * d_, nu = n_ctrl_();
        const bool free_y1 = !has_ic_;
        const int nz = nu + (free_y1 ? d_ : 0);
        // initial decision vector: zero control; initial state from the data-based guess (or the IC)
        vector_t z = vector_t::Zero(nz);
        if (free_y1) { z.segment(nu, d_) = initial_guess_().row(0).transpose(); }
        // objective of the starting point (zero control: the prior dynamics from the data-based initial
        // state). It seeds the best-iterate tracker -- x0 is a candidate like any accepted iterate, and the
        // callback below cannot see it -- and is the floor the guard further down tests against.
        const double J_start = objective_(z);
        if (std::isfinite(J_start)) { best_z_ = z; best_J_ = J_start; has_best_ = true; }
        // minimize the reduced objective over the control with BFGS
        control_objective problem {this};
        BFGS<Dynamic> optimizer(max_iter_, tol_, 1.0);

        // Wolfe line search: its curvature condition keeps the BFGS inverse Hessian positive
        // definite (descent directions), and its Armijo test rejects the divergent-cost surrogate.
        // MaxIter = 15 rather than the default 10. The bisection budget is what decides whether the
        // search returns a validated step or falls through with an unvalidated one, and on a badly
        // misspecified SB fit the fall-through step is large enough to blow the forward integration up:
        // the objective hits the divergent-cost sentinel, the envelope gradient goes identically zero,
        // and the OUTER BFGS then exits at iteration 0 with the estimate left at its starting guess.
        vector_t z_opt = optimizer.optimize(problem, z, WolfeLineSearch<15>(), best_iterate {this});
        
        /* Best-iterate guard.
        BFGS returns its LAST iterate, which is not necessarily its best. Two ways that bites here:
          - a trial control blows the forward integration up. There objective_ is the infinite sentinel
            and gradient_ is ZERO -- but a zero gradient is exactly how BFGS tests stationarity
            (error = ||grad|| <= tol), so the optimizer reads the blow-up as convergence, stops, and
            returns the divergent point, i.e. a non-finite trajectory reported as converged();
          - or the run merely ends on a hugely-worse-but-finite control, which no finiteness test catches.
        Both are covered by falling back to the best finite iterate the run visited (the same safeguard
        bs_ls_ode_nls applies to its shooting solve), recorded by the best_iterate callback.
        The trigger is the objective at the STARTING point, not the running best. Testing against the
        running best would fire on any solve whose final accepted step is a slight uphill move -- which a
        Wolfe search may legitimately take on the way to a stationary point -- and would then discard a
        converged solution for an earlier, worse-conditioned iterate. J_start (the u = 0 fit of the prior
        dynamics) is instead a fixed, meaningful floor: a solve that ends up worse than
        doing nothing at all has failed, whatever the reason. Written as !(J <= J_start) so a NaN objective
        also falls back. Convergence is then decided by an explicit test rather than by the iteration
        count, which cannot distinguish a genuine stationary exit from these. */
        if (has_best_ && !(objective_(z_opt) <= J_start)) { z_opt = best_z_; }
        // recover trajectory and diagnostics from the optimal control
        Y_ = forward_recover_(z_opt);
        if (has_ic_) { Y_.row(0) = y0_.transpose(); }
        // U_ stores the physical (unscaled) s stage values of every interval
        U_.resize(m_ - 1, sd);
        for (int t = 0; t < m_ - 1; ++t) { set_stage_block_(U_, t, expand_(z_opt, t)); }
        bound_mult_.setZero(m_, d_);   // the adjoint policy ignores state bounds
        objective_value_ = objective_(z_opt);   // consistent with z_opt after a possible fallback
        n_iter_ = optimizer.n_iter();
        // convergence = a finite trajectory AND a stationary gradient (the optimizer's own stopping
        // criterion, re-tested here at the returned point). An iteration count below the cap is on its
        // own evidence of neither.
        converged_ = Y_.allFinite() && gradient_(z_opt).norm() <= tol_;
        capture_dynamics_();
        compute_misfit_();
        flatten_();
        return f();
    }
    /* Full-space Gauss-Newton SQP over z = (Y, u)
    Keep the trajectory Y and the additive control u as decision variables, tie them by the discrete RK
    dynamics as equality constraints c_t = y_{t+1} - step(f + u_t)(y_t) = 0, and take constrained
    Gauss-Newton steps: each iteration solves the KKT saddle-point QP (exact quadratic objective Hessian
    + linearized constraints) and is globalized by an L1-merit backtracking line search. The constraint
    linearization uses the forward-mode step Jacobians d step/d y (flow) and d step/d u (control), so no
    adjoint sweep is involved.
    */
    const vector_t& fit_sqp_(double lambda) {
        fdapde_assert(lambda > 0 && d_ > 0 && m_ >= 2 && static_cast<bool>(engine_));
        if (has_ic_) { fdapde_assert(y0_.size() == d_); }
        lambda_ = lambda;
        coeff_valid_ = false;   // the expansion of the previous fit (see build_coefficients_)
        const int d = d_, m = m_, sd = s_ * d;
        const int nY = m * d, nU = (m - 1) * sd, n_primal = nY + nU;
        const int nC_dyn = (m - 1) * d, nC = nC_dyn + (has_ic_ ? d : 0);
        const vector_t b_quad = quad_weights_();   // stage weights of the control's L2 norm

        if (has_state_bounds_) { fdapde_assert(y_lb_.size() == d_ && y_ub_.size() == d_); }

        // decision variables: trajectory Y (data-based guess, IC-pinned first node), zero control u, zero
        // constraint multipliers -- the same starting-point basin as the adjoint policy.
        matrix_t Y = initial_guess_();
        if (has_ic_) { Y.row(0) = y0_.transpose(); }
        if (has_state_bounds_) {   // start feasible so the bound-preserving line search stays feasible
            for (int t = 0; t < m; ++t)
                for (int v = 0; v < d; ++v) Y(t, v) = std::min(std::max(Y(t, v), y_lb_(v)), y_ub_(v));
            if (has_ic_) { Y.row(0) = y0_.transpose(); }   // IC stays exact (assumed feasible)
        }
        matrix_t U = matrix_t::Zero(m - 1, sd);
        vector_t mu = vector_t::Zero(nC);
        bound_mult_.setZero(m, d);   // filled by sqp_subproblem_ when state bounds are active

        int it = 0;
        bool converged = false;
        // init SQP iterations
        for (; it < max_iter_; ++it) {
            // objective gradient (exact; J is quadratic in (Y, u)): data part 2 m (y - yobs), control part
            // 2 lambda dt u
            vector_t g = vector_t::Zero(n_primal);
            for (int t = 0; t < m; ++t) {
                for (int v = 0; v < d; ++v) {
                    if (mask_(t, v) != 0.0) { g(t * d + v) = 2.0 * (Y(t, v) - y_obs_(t, v)); }
                }
            }
            for (int t = 0; t < m - 1; ++t) {
                // penalty gradient, stage i weighted by b_i (the diagonal Gram matrix of the basis)
                for (int i = 0; i < s_; ++i) {
                    g.segment(nY + t * sd + i * d, d) =
                      2.0 * lambda_ * dt_(t) * b_quad[i] * U.block(t, i * d, 1, d).transpose();
                }
            }
            // constraints c and their Jacobian A: dynamics c_t = y_{t+1} - step(f + u_t)(y_t), with
            // d c_t/d y_{t+1} = I, d c_t/d y_t = -Flow_t, d c_t/d u_t = -B_t, plus the optional hard-IC
            // block c = y_0 - y0.
            vector_t c = vector_t::Zero(nC);
            std::vector<Eigen::Triplet<double>> A_trip;
            A_trip.reserve(static_cast<std::size_t>((m - 1) * d * (d + sd + 1) + (has_ic_ ? d : 0)));
            for (int t = 0; t < m - 1; ++t) {
                vector_t yc = Y.row(t).transpose();
                const matrix_t ut = stage_block_(U, t);
                auto [ynext, flow_t, b_t] = engine_.step_with_jacobians(time_(t), yc, dt_(t), ut);
                c.segment(t * d, d) = Y.row(t + 1).transpose() - ynext;
                const int r = t * d;
                for (int i = 0; i < d; ++i) {
                    A_trip.emplace_back(r + i, (t + 1) * d + i, 1.0);
                    for (int j = 0; j < d; ++j) { A_trip.emplace_back(r + i, t * d + j, -flow_t(i, j)); }
                    // d c_t/d u_t spans all s stage blocks of the interval
                    for (int j = 0; j < sd; ++j) { A_trip.emplace_back(r + i, nY + t * sd + j, -b_t(i, j)); }
                }
            }
            if (has_ic_) {
                for (int i = 0; i < d; ++i) { A_trip.emplace_back(nC_dyn + i, i, 1.0); }
                c.segment(nC_dyn, d) = Y.row(0).transpose() - y0_;
            }
            // convergence test. Unbounded: the KKT residual (stationarity ||g + A^T mu||_inf + feasibility).
            // Bounded: reduced stationarity is enforced by the active set, so test feasibility + a
            // vanishing step after the subproblem solve instead.
            double feas = (nC > 0) ? c.template lpNorm<Eigen::Infinity>() : 0.0;
            if (!has_state_bounds_) {
                sparse_matrix_t A(nC, n_primal);
                A.setFromTriplets(A_trip.begin(), A_trip.end());
                double stat = (g + A.transpose() * mu).template lpNorm<Eigen::Infinity>();
                if (std::max(stat, feas) < tol_) { converged = true; break; }
            }

            // solve the SQP subproblem: the plain KKT saddle point, or -- if state bounds are set -- the
            // bound-constrained QP  min 1/2 p'Hp + g'p  s.t.  A p = -c,  lb - Y <= p_Y <= ub - Y  (PDAS).
            vector_t p, mu_new;
            bool solve_ok = true;
            sqp_subproblem_(g, c, A_trip, n_primal, nC, Y, p, mu_new, solve_ok);
            if (!solve_ok) { break; }   // singular / NaN KKT: stop gracefully rather than abort

            if (has_state_bounds_ && std::max(feas, p.template lpNorm<Eigen::Infinity>()) < tol_) {
                converged = true; break;
            }

            // reshape the primal step into node/interval blocks
            matrix_t dY(m, d), dU(m - 1, sd);
            for (int t = 0; t < m; ++t) { dY.row(t) = p.segment(t * d, d).transpose(); }
            for (int t = 0; t < m - 1; ++t) { dU.row(t) = p.segment(nY + t * sd, sd).transpose(); }

            // L1-merit backtracking line search: M(z; rho) = J(z) + rho ||c(z)||_1 with rho > ||mu_new||_inf
            // guarantees a descent direction (A p = -c linearizes the constraint away).
            double c1 = c.template lpNorm<1>();
            double rho = 1.5 * mu_new.template lpNorm<Eigen::Infinity>() + 1e-6;
            double M0 = objective_YU_(Y, U) + rho * c1;
            double DM = g.dot(p) - rho * c1;
            double alpha = 1.0;
            const double eta = 1e-4, shrink = 0.5;
            bool accepted = false;
            for (int ls = 0; ls < 40; ++ls) {
                matrix_t Yt = Y + alpha * dY, Ut = U + alpha * dU;
                double Mt = std::numeric_limits<double>::infinity();
                if (Yt.allFinite() && Ut.allFinite()) {
                    vector_t ct = constraints_(Yt, Ut);
                    if (ct.allFinite()) { Mt = objective_YU_(Yt, Ut) + rho * ct.template lpNorm<1>(); }
                }
                if (Mt <= M0 + eta * alpha * DM) { Y = Yt; U = Ut; accepted = true; break; }
                alpha *= shrink;
            }
            if (!accepted) { Y += alpha * dY; U += alpha * dU; }   // damped step to keep progressing
            mu = mu_new;
        }
        n_iter_ = it;
        converged_ = converged;

        // recover trajectory and diagnostics
        Y_ = Y;
        if (has_ic_) { Y_.row(0) = y0_.transpose(); }
        U_ = U;   // the additive control itself (the decision variable)
        objective_value_ = objective_YU_(Y_, U);
        capture_dynamics_();
        compute_misfit_();
        flatten_();
        return f();
    }

    /* Solve one SQP subproblem. 
    Without state bounds: the KKT saddle [[H, A^T],[A,0]] [p; mu] = [-g; -c]
    (H = the exact, diagonal objective Hessian) -- one sparse solve, identical to the unconstrained SQP.
    With state bounds: the same objective/dynamics plus the box  lb - Y <= p_Y <= ub - Y, solved by a
    primal-dual active-set loop -- append the active-bound rows e_k^T p = bound - Y_k to the KKT, then
    ADD a bound whose step violates it and RELEASE an active bound whose multiplier has the wrong sign
    (reduced gradient = -nu: a lower bound stays iff nu <= 0, an upper bound iff nu >= 0). The active set
    at convergence is exactly the set of boundary arcs. Sets ok=false on a failed factorization so the
    caller bails gracefully instead of asserting.
    */
    void sqp_subproblem_(const vector_t& g, const vector_t& c,
                         const std::vector<Eigen::Triplet<double>>& A_trip, int n_primal, int nC,
                         const matrix_t& Y, vector_t& p, vector_t& mu_new, bool& ok) {
        const int d = d_, m = m_, nY = m * d, sd = s_ * d;
        const vector_t b_quad = quad_weights_();   // stage weights of the penalty Hessian
        std::vector<int> st(nY, 0);   // per (node, component): 0 free, -1 pinned at lb, +1 pinned at ub
        for (int sweep = 0; sweep < 6 * nY + 20; ++sweep) {
            std::vector<int> W; std::vector<double> bW;
            if (has_state_bounds_) {
                for (int t = 0; t < m; ++t) {
                    if (has_ic_ && t == 0) { continue; }   // node 0 is pinned by the IC equality already
                    for (int v = 0; v < d; ++v) {
                        const int k = t * d + v;
                        if (st[k] == -1) { W.push_back(k); bW.push_back(y_lb_(v) - Y(t, v)); }
                        else if (st[k] == 1) { W.push_back(k); bW.push_back(y_ub_(v) - Y(t, v)); }
                    }
                }
            }
            const int nW = static_cast<int>(W.size()), N = n_primal + nC + nW;
            std::vector<Eigen::Triplet<double>> K_trip;
            K_trip.reserve(static_cast<std::size_t>(n_primal + 2 * A_trip.size() + 2 * nW));
            // H (exact objective Hessian): 2*mask on the Y-block, 2*lambda*dt on the u-block
            for (int t = 0; t < m; ++t)
                for (int v = 0; v < d; ++v)
                    if (mask_(t, v) != 0.0) { K_trip.emplace_back(t * d + v, t * d + v, 2.0 * mask_(t, v)); }
            // penalty Hessian: diagonal, stage i scaled by the quadrature weight b_i
            for (int t = 0; t < m - 1; ++t)
                for (int i = 0; i < s_; ++i)
                    for (int v = 0; v < d; ++v)
                        K_trip.emplace_back(
                          nY + t * sd + i * d + v, nY + t * sd + i * d + v, 2.0 * lambda_ * dt_(t) * b_quad[i]);
            for (const auto& tr : A_trip) {                                    // A and A^T
                K_trip.emplace_back(n_primal + tr.row(), tr.col(), tr.value());
                K_trip.emplace_back(tr.col(), n_primal + tr.row(), tr.value());
            }
            for (int w = 0; w < nW; ++w) {                                     // active-bound rows e_k and e_k^T
                K_trip.emplace_back(n_primal + nC + w, W[w], 1.0);
                K_trip.emplace_back(W[w], n_primal + nC + w, 1.0);
            }
            sparse_matrix_t K(N, N); K.setFromTriplets(K_trip.begin(), K_trip.end()); K.makeCompressed();
            Eigen::SparseLU<sparse_matrix_t> lu; lu.compute(K);
            if (lu.info() != Eigen::Success) { ok = false; return; }
            vector_t rhs(N); rhs.head(n_primal) = -g; rhs.segment(n_primal, nC) = -c;
            for (int w = 0; w < nW; ++w) { rhs(n_primal + nC + w) = bW[w]; }
            vector_t sol = lu.solve(rhs);
            if (lu.info() != Eigen::Success) { ok = false; return; }
            p = sol.head(n_primal);
            mu_new = sol.segment(n_primal, nC);
            if (!has_state_bounds_) { ok = true; return; }
            const vector_t nu = sol.segment(n_primal + nC, nW);
            // record the active-bound multipliers on the (node, component) grid: they are the dual
            // variables of the state box, and an inverse solver needs them to augment the node source
            // (the constrained envelope theorem -- the bounds constrain Y, which depends on theta).
            bound_mult_.setZero(m_, d_);
            for (int w = 0; w < nW; ++w) { bound_mult_(W[w] / d, W[w] % d) = nu(w); }
            // 1) activate the inactive bounds the step violates
            bool changed = false;
            for (int t = 0; t < m; ++t) {
                if (has_ic_ && t == 0) { continue; }
                for (int v = 0; v < d; ++v) {
                    const int k = t * d + v;
                    if (st[k] != 0) { continue; }
                    const double lo = y_lb_(v) - Y(t, v), hi = y_ub_(v) - Y(t, v);
                    if (p(k) < lo - 1e-10) { st[k] = -1; changed = true; }
                    else if (p(k) > hi + 1e-10) { st[k] = 1; changed = true; }
                }
            }
            if (changed) { continue; }
            // 2) release one active bound whose multiplier has the wrong sign
            for (int w = 0; w < nW; ++w) {
                const int k = W[w];
                if (st[k] == -1 && nu(w) > 1e-9) { st[k] = 0; changed = true; break; }
                if (st[k] == 1 && nu(w) < -1e-9) { st[k] = 0; changed = true; break; }
            }
            if (!changed) { ok = true; return; }   // KKT + bound complementarity satisfied
        }
        ok = true;   // reached the sweep cap (rare): return the last computed step
    }

    // objective J = SSE(observed) + lambda sum_t dt_t ||u_t||^2 as a function of the full-space (Y, u)
    double objective_YU_(const matrix_t& Y, const matrix_t& U) const {
        const vector_t b_quad = quad_weights_();
        double J = 0;
        for (int t = 0; t < m_; ++t) {
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { J += std::pow(Y(t, v) - y_obs_(t, v), 2); }
            }
        }
        // penalty = lambda * int ||u||^2 = lambda * sum_t dt_t sum_i b_i ||u_{t,i}||^2 (exact, not quadrature)
        for (int t = 0; t < m_ - 1; ++t) {
            for (int i = 0; i < s_; ++i) {
                J += lambda_ * dt_(t) * b_quad[i] * U.block(t, i * d_, 1, d_).squaredNorm();
            }
        }
        return J;
    }
    // equality-constraint residual c(Y, u): dynamics defects c_t = y_{t+1} - step(f + u_t)(y_t) followed
    // by the optional hard-IC defect y_0 - y0 (used to re-evaluate the merit during the line search)
    vector_t constraints_(const matrix_t& Y, const matrix_t& U) const {
        const int d = d_, nC_dyn = (m_ - 1) * d;
        vector_t c(nC_dyn + (has_ic_ ? d : 0));
        for (int t = 0; t < m_ - 1; ++t) {
            vector_t yc = Y.row(t).transpose();
            const matrix_t ut = stage_block_(U, t);
            vector_t ynext = engine_.step(time_(t), yc, dt_(t), ut);
            c.segment(t * d, d) = Y.row(t + 1).transpose() - ynext;
        }
        if (has_ic_) { c.segment(nC_dyn, d) = Y.row(0).transpose() - y0_; }
        return c;
    }

    // GCV utilities
    
    /* Gauss-Newton Hessian of the penalized objective
    
    The penalty lambda sum_t dt_t u_t^T W u_t (W = diag(b_i) x I, the exact Gram matrix of the control
    basis) must be expressed as a function of Y, with u_t the additive control that was fitted -- i.e. the
    control implied by consecutive nodes through the FORCED dynamics y_{t+1} = step(f + u_t)(y_t).

    With a STAGE-WISE control that relation no longer determines u_t: b_t = d y_{t+1}/d u_t is d x (s*d),
    so s*d control degrees of freedom produce only d outputs and many controls join the same pair of nodes.
    The selection consistent with the fit is the one the optimizer actually makes -- among the controls
    reaching y_{t+1}, the one of least penalty -- i.e. the W-weighted minimum-norm solution, whose
    linearization is the W-weighted pseudoinverse
        P = W^{-1} b_t^T (b_t W^{-1} b_t^T)^{-1}          (s*d x d)
        d u_t/d y_{t+1} =  P            (=: G1)
        d u_t/d y_t     = -P flow_t     (=: G0),
    so the interval's Gauss-Newton block is cw [G0;G1]^T W [G0;G1] with cw = lambda dt_t. For a 1-stage
    scheme W = I and P = b_t^{-1}, recovering the square-inverse form this had before.
    */
    void assemble_(sparse_matrix_t& H) const {
        const int N = m_ * d_;
        const vector_t b_quad = quad_weights_();
        std::vector<Eigen::Triplet<double>> triplets;
        auto is_fixed = [&](int node) { return has_ic_ && node == 0; };
        auto add_block = [&](int rn, int cn, const matrix_t& blk) {
            if (is_fixed(rn) || is_fixed(cn)) { return; }
            for (int r = 0; r < d_; ++r) {
                for (int c = 0; c < d_; ++c) {
                    if (blk(r, c) != 0.0) { triplets.emplace_back(rn * d_ + r, cn * d_ + c, blk(r, c)); }
                }
            }
        };
        // data term (Hessian of sum mask (y - yobs)^2 in the factor-1 convention: identity on observed dofs)
        for (int t = 0; t < m_; ++t) {
            if (is_fixed(t)) { continue; }
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { triplets.emplace_back(t * d_ + v, t * d_ + v, 1.0); }
            }
        }
        // penalty term: each interval couples nodes t and t+1 through the forced control u_t = u_t(y_t, y_{t+1})
        fdapde_assert(static_cast<int>(jac_B_.size()) == m_ - 1 && "no fit has captured its dynamics");
        for (int t = 0; t < m_ - 1; ++t) {
            const double cw = lambda_ * dt_(t);
            // forced step Jacobians at the fitted control, captured by the fit (see capture_dynamics_):
            // jac_flow_ = d y_{t+1}/d y_t, jac_B_ = d y_{t+1}/d u_t
            const matrix_t& flow_t = jac_flow_[t];
            const matrix_t& B_t = jac_B_[t];
            // W-weighted pseudoinverse of the (d x s*d) control Jacobian; for s = 1 this is B_t^{-1}
            vector_t winv(sd_());
            for (int i = 0; i < s_; ++i) { winv.segment(i * d_, d_).setConstant(1.0 / b_quad[i]); }
            const matrix_t BWi = B_t * winv.asDiagonal();                // b_t W^{-1}      (d x s*d)
            const matrix_t P = winv.asDiagonal() * B_t.transpose() *
                               (BWi * B_t.transpose()).inverse();        // P               (s*d x d)
            const matrix_t G0 = -P * flow_t;                             // d u_t/d y_t
            const matrix_t& G1 = P;                                      // d u_t/d y_{t+1}
            // the blocks carry the W metric of the penalty: [G0;G1]^T W [G0;G1]
            const matrix_t WG0 = winv.asDiagonal().inverse() * G0, WG1 = winv.asDiagonal().inverse() * G1;
            add_block(t,     t,     cw * (G0.transpose() * WG0));
            add_block(t,     t + 1, cw * (G0.transpose() * WG1));
            add_block(t + 1, t,     cw * (G1.transpose() * WG0));
            add_block(t + 1, t + 1, cw * (G1.transpose() * WG1));
        }
        if (has_ic_) {
            for (int v = 0; v < d_; ++v) { triplets.emplace_back(v, v, 1.0); }
        }
        H.resize(N, N);
        H.setFromTriplets(triplets.begin(), triplets.end());
        H.makeCompressed();
        return;
    }


    // model and discretization: the type-erased control-aware engine (bundles the prior field and the
    // RK integrator; both the stage count and the system dimension are erased inside it)
    any_controlled_ode_solver engine_;
    
    int max_iter_ = 500; // BFGS iteration cap
    double tol_ = 1e-8;
    bool has_ic_ = false;
    vector_t y0_;

    // optional per-component state box constraints y_lb_ <= y_{t,:} <= y_ub_ (d-vectors, applied at every
    // time node; use +/- infinity to leave a side free). Honoured by the SQP policy via a primal-dual
    // active-set inner solve; the adjoint policy ignores them.
    bool has_state_bounds_ = false;
    vector_t y_lb_, y_ub_;

    // data
    /* The mesh. The m nodes where observations live, where the control is parameterised, and on which
    the ODE is stepped -- one mesh for all three (see INTEGRATION_REFINEMENT.md for the integration mesh
    that used to be separate from it). */
    vector_t time_;                    // m nodes (also the observation mesh, via mask_)
    int m_ = 0, d_ = 0, n_obs_ = 0;    // n. time instants, n. components, n. observed scalars
    int s_ = 1;                        // RK stages of the scheme = degrees of freedom of the control per interval
    int degree_ = 1;                   // degree of the declared trajectory space (== s_)
    /* What discretize extracts from the trajectory space (C0, degree p) and the control space (broken,
    degree p-1): no space is kept, as in fe_ls_elliptic. The evaluation handles hold copies of the spaces,
    which point at the caller's mesh. */
    int n_dofs_ = 0, n_ctrl_dofs_ = 0;
    vector_t mesh_nodes_;
    std::vector<cell_basis_t> stage_basis_, ctrl_stage_basis_;   // per cell (see sample_at_stages_)
    std::function<sparse_matrix_t(const matrix_t& locs)> point_eval_, ctrl_point_eval_;
    // names of the state components, when the rhs form supplied them; empty means a positional response
    std::vector<std::string> component_names_;
    matrix_t y_obs_, y_fill_, mask_;   // observations, NaN-filled copy, observation mask

    // one-slot cache of the controlled forward pass (see forward_recover_): the recovered trajectory and the
    // per-interval forward-mode Jacobians at fwd_z_, shared by objective_ / gradient_ across the BFGS solve.
    // diagonal preconditioner of the reduced solve (see ctrl_scale_ / build_ctrl_scale_): per decision block,
    // the control scale s and penalty weight w = 2 lambda dt_t b_i / s^2, rebuilt at each fit_adjoint_.
    std::vector<double> ctrl_s_, ctrl_w_;

    // best finite iterate seen during the reduced solve, and its objective: the fallback used by the
    // divergence guard in fit_adjoint_ when BFGS returns a blown-up point as its optimum.
    vector_t best_z_;
    double best_J_ = 0;
    bool has_best_ = false;

    // whether f_ holds the expansion of the current fit (see build_coefficients_)
    mutable bool coeff_valid_ = false;
    // per-interval dynamics of the current fit (see capture_dynamics_): stage values, d y_{t+1}/d y_t, d y_{t+1}/d u_t
    std::vector<matrix_t> stage_values_, jac_flow_, jac_B_;

    vector_t fwd_z_;                     // control at which the cache was built
    matrix_t fwd_Y_;                     // recovered trajectory (m x d)
    std::vector<matrix_t> fwd_flow_;     // d y_{t+1}/d y_t per interval
    std::vector<matrix_t> fwd_B_;        // d y_{t+1}/d u_t per interval
    bool fwd_valid_ = false;

    // results
    double lambda_ = -1;
    matrix_t Y_, misfit_;         // fitted trajectory (m x d) and ODE-residual misfit ((m-1) x d)
    // the additive control, ((m-1) x (s*d)): row t holds the s stage values of the control polynomial on
    // interval t, stage-major (block i = the control at node c_i). The actual decision variable.
    matrix_t U_;
    matrix_t bound_mult_;         // state-box dual variables (m x d), zero where inactive
    mutable vector_t f_;   // expansion coefficients in the trajectory space, dof-major (see build_coefficients_)
    mutable vector_t uc_;  // expansion coefficients of the control in its broken space, dof-major
    vector_t fn_, g_, y_;  // flattened nodal trajectory / misfit / response
    double objective_value_ = 0;
    int n_iter_ = 0;
    bool converged_ = false;
};

}   // namespace internals

// Time-stepping ODE-penalty solver API: penalty descriptor wrapping the vector field, the
// time-integration scheme (Butcher tableau) and an optional initial condition. The descriptor is not
// templated on the ODE system dimension: the dimension is deduced from the field functor (a fixed-size
// return type -> static Dim, VectorXd -> Dynamic) as a constructor-local constexpr, used only to build
// the typed field/engine, then erased away. So the descriptor, the solver and the model wrapper that
// consume it are all free of both the stage-count and the system-dimension template parameters.
struct bs_ls_ode {
    using solver_t = internals::bs_ls_ode;
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    struct penalty_packet {
        any_controlled_ode_solver engine_;
        std::vector<std::string> names_;   // component names of the rhs form; empty for a plain functor
        vector_t ic_;
        bool has_ic_ = false;
        int degree_ = 1;   // degree of the declared trajectory space == stage count of the scheme
        const BsSpace<Triangulation<1, 1>>* space_ = nullptr;   // the trajectory space, read by discretize
        int max_iter_ = 500;
        double tol_ = 1e-8;
       public:
        penalty_packet(
          any_controlled_ode_solver engine, int degree, const BsSpace<Triangulation<1, 1>>* space, int max_iter,
          double tol) :
            engine_(std::move(engine)), degree_(degree), space_(space), max_iter_(max_iter), tol_(tol) { }
        penalty_packet(
          any_controlled_ode_solver engine, int degree, const BsSpace<Triangulation<1, 1>>* space, vector_t ic,
          int max_iter, double tol) :
            engine_(std::move(engine)), ic_(std::move(ic)), has_ic_(true), degree_(degree), space_(space),
            max_iter_(max_iter), tol_(tol) { }
        // observers
        int degree() const { return degree_; }
        const BsSpace<Triangulation<1, 1>>* space() const { return space_; }
        const any_controlled_ode_solver& engine() const { return engine_; }
        const std::vector<std::string>& component_names() const { return names_; }
        void set_component_names(std::vector<std::string> names) { names_ = std::move(names); }
        const vector_t& ic() const { return ic_; }
        bool has_ic() const { return has_ic_; }
        int max_iter() const { return max_iter_; }
        double tol() const { return tol_; }
    };
    // bundle the field and the freshly-built integrator into the erased control-aware engine. Dim is the
    // field's static dimension (fixed-size return -> static, VectorXd -> Dynamic), deduced here and used
    // together with the functor type Field only to build the typed controlled_ode_solver<Stages, Dim, Field>; neither
    // escapes this function (both are hidden behind the erased any_controlled_ode_solver).
    template <typename Field, int Stages>
    static any_controlled_ode_solver engine_from_tableau_(const Field& field, const ButcherTableau<Stages>& tableau) {
        constexpr int Dim = ode_rhs_dim_v<Field>;
        return any_controlled_ode_solver(
          controlled_ode_solver<Stages, Dim, Field>(ode_rhs_field<Dim, Field>(field), RKIntegrator(tableau)));
    }
    /* The trajectory space fixes the scheme. A degree-p trajectory is realised by a p-stage collocation
    method, and the method must be a GAUSS one: symplecticity is what makes the discrete adjoint coincide
    with the same scheme run backward, which is what lets the stage costates be read as collocation data
    (see RKIntegrator::adjoint_step_with_stages). So the degree selects the tableau, and no other tableau is
    reachable -- there is deliberately no way to hand this solver a Radau, Lobatto or explicit scheme. */
    template <typename Field> static any_controlled_ode_solver engine_from_degree_(const Field& field, int degree) {
        switch (degree) {
        case 1: return engine_from_tableau_(field, ode_schemes::implicit_midpoint());
        case 2: return engine_from_tableau_(field, ode_schemes::gauss_legendre_2());
        case 3: return engine_from_tableau_(field, ode_schemes::gauss_legendre_3());
        case 4: return engine_from_tableau_(field, ode_schemes::gauss_legendre_4());
        }
        fdapde_assert(
          false && "trajectory space degree must be 1, 2, 3 or 4 (the Gauss-Legendre schemes available)");
        return any_controlled_ode_solver();
    }
   public:
    /* Construct from the prior dynamics and the TRAJECTORY SPACE the fit is expanded in.
    The space is the C0 spline space of degree p over the time mesh, BsSpace(mesh, p, std::vector<int>(n, p)).
    Its degree is the only discretization choice and determines everything else: the scheme is the p-stage
    Gauss collocation method, and the control -- the trajectory's derivative -- is the broken space of
    degree p-1 on the same mesh.

    The descriptor refers to the space, as fe_ls_elliptic's forms refer to theirs, and the solver reads it
    once, in discretize, keeping nothing of it: the space must be alive then, and no longer. The mesh must
    outlive the solver, whose evaluation handles point at it, and must be the one the data lives on (checked
    in check_mesh_). Building the GeoFrame and the space over one Triangulation, as sr.cpp does on the FE
    side, satisfies both at once. */
    template <typename Field>
        requires(!is_ode_system_v<Field>)
    bs_ls_ode(const Field& field, const BsSpace<Triangulation<1, 1>>& Vh, int max_iter = 500, double tol = 1e-8) :
        penalty_(engine_from_degree_(field, Vh.order()), Vh.order(), std::addressof(Vh), max_iter, tol) { }
    template <typename Field>
        requires(!is_ode_system_v<Field>)
    bs_ls_ode(
      const Field& field, const BsSpace<Triangulation<1, 1>>& Vh, const vector_t& ic, int max_iter = 500,
      double tol = 1e-8) :
        penalty_(engine_from_degree_(field, Vh.order()), Vh.order(), std::addressof(Vh), ic, max_iter, tol) { }
    /* Construct from a system in strong form. Its unknowns are declared over the trajectory space, so the space
    is not passed again, and their names bind the response by name. The space must be alive at discretize. */
    template <typename... Eqs>
    explicit bs_ls_ode(const ode_system<Eqs...>& sys, int max_iter = 500, double tol = 1e-8) :
        penalty_(
          engine_from_degree_(sys, sys.function_space().order()), sys.function_space().order(),
          std::addressof(sys.function_space()), max_iter, tol) {
        penalty_.set_component_names(sys.component_names());
    }
    template <typename... Eqs>
    bs_ls_ode(const ode_system<Eqs...>& sys, const vector_t& ic, int max_iter = 500, double tol = 1e-8) :
        penalty_(
          engine_from_degree_(sys, sys.function_space().order()), sys.function_space().order(),
          std::addressof(sys.function_space()), ic, max_iter, tol) {
        penalty_.set_component_names(sys.component_names());
    }
    const penalty_packet& get() const { return penalty_; }
   private:
    penalty_packet penalty_;
};

}   // namespace fdapde

#endif   // __BS_LS_ODE_SOLVER_H__
