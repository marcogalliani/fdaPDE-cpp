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

#ifndef __TS_LS_ODE_TRACKING_SOLVER_H__
#define __TS_LS_ODE_TRACKING_SOLVER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// Parameter-aware engine: the theta-parameterized dynamics f(t, y, theta) together with the RK scheme.
// It is NOT the engine the inner solver runs on -- it *produces* one. For a given theta it binds the
// parameter away (theta_bound_rhs) and hands back an ordinary rk_engine, so the whole forward stack
// (ts_ls_ode, its two fit policies, the integrator) is reused untouched. On top of that it exposes the
// one extra quantity the inverse problem needs: d step / d theta on the control-forced dynamics.
template <int Stages, int Dim, typename F> struct param_rk_engine {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    param_rk_engine() = default;
    param_rk_engine(F field, RKIntegrator<Stages> integrator, vector_t theta) :
        field_(std::move(field)), integrator_(std::move(integrator)), theta_(std::move(theta)) { }

    void set_theta(const vector_t& theta) { theta_ = theta; }
    const vector_t& theta() const { return theta_; }
    int n_params() const { return theta_.size(); }

    // the plain (theta-bound) control-aware engine the inner solver runs on
    rk_engine<Stages, Dim> engine() const {
        return rk_engine<Stages, Dim>(ode_rhs_field<Dim>(bound_()), integrator_);
    }
    // d step / d theta at (t, y, dt) on the dynamics forced by the constant control u (d x n_theta).
    // u is theta-independent, so the parameter Jacobian fed to the integrator is that of the unforced
    // parametric field, while the stage system itself is built on the forced one.
    matrix_t param_jacobian(double t, const vector_t& y, double dt, const vector_t& u) const {
        ode_rhs_field<Dim> bound(bound_());
        return integrator_.step_param_jacobian(
          bound + u, t, y, dt,
          [this](double tt, const vector_t& yy) { return df_dtheta_(tt, yy); }, theta_.size());
    }

   private:
    theta_bound_rhs<F> bound_() const { return theta_bound_rhs<F> {field_, theta_}; }
    // analytic df/dtheta when the field supplies one, central finite differences otherwise
    matrix_t df_dtheta_(double t, const vector_t& y) const {
        if constexpr (has_param_jacobian<F>) {
            return field_.df_dtheta(t, y, theta_);
        } else {
            return param_jacobian_fd(field_, t, y, theta_);
        }
    }

    F field_;
    RKIntegrator<Stages> integrator_;
    vector_t theta_;
};

// The handle the tracking solver keeps: closures over one shared param_rk_engine, so Stages / Dim / the
// field type are all erased without introducing a second type-erasure interface. engine() snapshots the
// engine at the currently bound theta into the plain any_rk_engine the base solver consumes.
struct param_engine_handle {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    std::function<void(const vector_t&)> set_theta;
    std::function<any_rk_engine()> engine;
    std::function<matrix_t(double, const vector_t&, double, const vector_t&)> param_jacobian;
    int n_params = 0;

    explicit operator bool() const { return static_cast<bool>(engine); }
};

template <int Stages, int Dim, typename F>
param_engine_handle make_param_engine_handle(
  F field, RKIntegrator<Stages> integrator, Eigen::Matrix<double, Dynamic, 1> theta) {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    auto eng = std::make_shared<param_rk_engine<Stages, Dim, F>>(
      std::move(field), std::move(integrator), std::move(theta));
    param_engine_handle h;
    h.set_theta = [eng](const vector_t& th) { eng->set_theta(th); };
    // erase's templated converting constructor accepts the concrete engine: it implements the whole
    // IRkEngine surface, so no new erased interface is needed.
    h.engine = [eng]() { return any_rk_engine(eng->engine()); };
    h.param_jacobian = [eng](double t, const vector_t& y, double dt, const vector_t& u) {
        return eng->param_jacobian(t, y, dt, u);
    };
    h.n_params = eng->n_params();
    return h;
}

// Tracking inverse solver: estimates the ODE parameters theta of a parameterized prior dynamics
// y' = f(t, y, theta) + u(t) from the same data the forward solver smooths.
//
// Following the tracking estimator (Brunel & Clairon), the outer criterion is the *same* functional as
// the inner one,
//
//   H(theta) = min_u J(u, theta),        theta* = argmin_theta H(theta),
//
// so the whole problem is the joint minimization min_{u, theta} J(u, theta) organized as
// min_theta min_u J(u, theta). It derives from ts_ls_ode and uses the inherited fit() as its inner
// solver: the state and the control never leave the forward solver.
//
// Because the outer criterion coincides with the inner one, the inner optimality condition dJ/du = 0
// makes the total derivative collapse onto the partial one (envelope theorem):
//
//   grad_theta H = d/dtheta J(u, theta) | u = u*(theta),
//
// and since the penalty lambda sum dt ||u||^2 carries no theta, only the data term survives. So NO inner
// sensitivity system is solved (that is the parameter-cascading route): the trajectory sensitivity at
// frozen control is accumulated in a single backward adjoint sweep,
//
//   p = s_{m-1};  for t = m-2..0:  grad += Theta_t^T p;  p = s_t + Flow_t^T p,
//
// with s_t = 2 m_t (y_t - y^obs_t) the node source, Flow_t^T p supplied by the engine's existing
// adjoint_step, and Theta_t = d step / d theta the one new quantity (RKIntegrator::step_param_jacobian).
// When state bounds are active the node source is augmented by their dual variables, which is what the
// constrained envelope theorem prescribes (the box constrains Y, and Y depends on theta).
class ts_ls_ode_tracking : public ts_ls_ode {
   public:
    using ts_ls_ode::fit;   // the inner solver, inherited unchanged
    using Base = ts_ls_ode;

    ts_ls_ode_tracking() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    ts_ls_ode_tracking(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty);
        analyze_data(formula, gf);
    }

    // discretize a parametric penalty: the base takes the engine at theta0 (so a plain forward fit()
    // works immediately), and we keep the handle to rebind theta during the outer solve.
    template <typename Penalty> void discretize(Penalty&& penalty) {
        Base::discretize(penalty);
        handle_ = penalty.handle();
        theta_ = penalty.theta0();
        fdapde_assert(static_cast<bool>(handle_) && theta_.size() == handle_.n_params);
        invalidate_cache_();
    }

    // --- configuration -------------------------------------------------
    // inner fit policy. By default the reduced adjoint solve, automatically switched to SQP when state
    // bounds are set (only the SQP policy honours them).
    void set_inner_policy(fit_policy policy) {
        inner_policy_ = policy;
        policy_override_ = true;
    }
    void use_default_inner_policy() { policy_override_ = false; }
    void set_outer_options(int max_iter, double tol) {
        fdapde_assert(max_iter > 0 && tol > 0);
        outer_max_iter_ = max_iter;
        outer_tol_ = tol;
    }
    // d y_0 / d theta (d x n_theta), only meaningful under a hard initial condition that itself depends
    // on theta. Absent (the default) the first node contributes nothing: with no hard IC the initial
    // state is an inner decision variable, so the envelope theorem freezes it too.
    void set_ic_jacobian(std::function<matrix_t(const vector_t&)> ic_jacobian) {
        ic_jacobian_ = std::move(ic_jacobian);
    }

    // --- outer solve ---------------------------------------------------
    // minimize H(theta) = min_u J(u, theta) over theta by BFGS on the envelope gradient. On return the
    // inherited forward state (trajectory, control, f(), ...) is the inner fit at the estimated theta.
    const vector_t& solve(double lambda, const vector_t& theta0) {
        fdapde_assert(lambda > 0 && static_cast<bool>(handle_));
        fdapde_assert(theta0.size() == handle_.n_params);
        lambda_outer_ = lambda;
        inner_failures_ = 0;
        invalidate_cache_();
        tracking_objective problem {this};
        BFGS<Dynamic> optimizer(outer_max_iter_, outer_tol_, 1.0);
        // Wolfe line search, as in the inner adjoint solve: the curvature condition keeps the BFGS
        // inverse Hessian positive definite and the Armijo test rejects the divergent-cost surrogate.
        vector_t theta_opt = optimizer.optimize(problem, theta0, WolfeLineSearch());
        theta_ = theta_opt;
        outer_value_ = optimizer.value();
        outer_n_iter_ = optimizer.n_iter();
        outer_converged_ = (outer_n_iter_ < outer_max_iter_);
        // leave the forward state consistent with the estimate
        invalidate_cache_();
        inner_solve_(theta_);
        return theta_;
    }
    const vector_t& solve(double lambda) { return solve(lambda, theta_); }

    // --- observers -----------------------------------------------------
    const vector_t& theta() const { return theta_; }            // parameter estimate
    int n_params() const { return handle_.n_params; }
    double outer_objective() const { return outer_value_; }     // H(theta*)
    int outer_n_iter() const { return outer_n_iter_; }
    bool outer_converged() const { return outer_converged_; }
    // number of outer evaluations whose inner solve did not meet its tolerance. The envelope gradient is
    // only valid at an inner optimum, so a nonzero count means the outer gradient was inconsistent.
    int n_inner_failures() const { return inner_failures_; }
    fit_policy inner_policy() const {
        return policy_override_ ? inner_policy_ : (has_state_bounds() ? fit_policy::sqp : fit_policy::adjoint);
    }

    // H(theta) and its envelope gradient at an explicit lambda, exposed for diagnostics and for the
    // finite-difference cross-checks. Memoization still collapses an objective/gradient pair at the same
    // (lambda, theta) onto a single inner solve.
    double outer_objective_at(double lambda, const vector_t& theta) {
        set_lambda_(lambda);
        return H_(theta);
    }
    vector_t outer_gradient_at(double lambda, const vector_t& theta) {
        set_lambda_(lambda);
        return grad_H_(theta);
    }

   private:
    // run the inner solve at theta, memoized: BFGS asks for the objective and the gradient at the same
    // point, and one inner solve serves both.
    void inner_solve_(const vector_t& theta) {
        if (cache_valid_ && theta_cache_.size() == theta.size() && theta_cache_ == theta) { return; }
        handle_.set_theta(theta);
        engine_ = handle_.engine();   // rebind the inherited engine to the new parameters
        fit(lambda_outer_, inner_policy());
        theta_cache_ = theta;
        cache_valid_ = true;
        if (!converged()) { ++inner_failures_; }
        return;
    }
    void invalidate_cache_() {
        cache_valid_ = false;
        theta_cache_.resize(0);
        return;
    }
    void set_lambda_(double lambda) {
        fdapde_assert(lambda > 0);
        if (lambda != lambda_outer_) {
            lambda_outer_ = lambda;
            invalidate_cache_();   // the cached inner solve belongs to the old lambda
        }
        return;
    }

    // H(theta): the inner optimal value. A control that blows the forward integration up yields a large
    // finite cost so the outer line search backtracks out of the divergent region.
    double H_(const vector_t& theta) {
        inner_solve_(theta);
        double v = objective();
        if (!std::isfinite(v) || !Y_.allFinite()) { return divergent_outer_cost_; }
        return v;
    }

    // envelope gradient: a single backward adjoint sweep at frozen control.
    vector_t grad_H_(const vector_t& theta) {
        inner_solve_(theta);
        const int n_theta = handle_.n_params;
        vector_t grad = vector_t::Zero(n_theta);
        // divergent inner solve: report a finite (zero) gradient, the large objective drives the line
        // search back on its own.
        if (!Y_.allFinite() || !U_.allFinite()) { return grad; }
        const bool bounded = (bound_mult_.rows() == m_ && bound_mult_.cols() == d_);
        // node source dJ/dy_t, augmented by the state-box duals: with an active box the inner problem is
        // constrained by a set that moves with theta, and the constrained envelope theorem adds exactly
        // this multiplier term.
        auto source = [&](int t) {
            vector_t s = vector_t::Zero(d_);
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { s(v) = 2.0 * (Y_(t, v) - y_obs_(t, v)); }
            }
            if (bounded) { s += bound_mult_.row(t).transpose(); }
            return s;
        };
        vector_t p = source(m_ - 1);
        for (int t = m_ - 2; t >= 0; --t) {
            vector_t yc = Y_.row(t).transpose();
            vector_t ut = U_.row(t).transpose();
            // d y_{t+1}/d theta on the forced dynamics; p is the costate p_{t+1}
            matrix_t Theta = handle_.param_jacobian(time_(t), yc, dt_(t), ut);
            grad.noalias() += Theta.transpose() * p;
            // Flow_t^T p, from the engine's existing discrete adjoint of one step
            vector_t p_prop = engine_.adjoint_step(time_(t), yc, dt_(t), p, ut).first;
            p = source(t) + p_prop;
        }
        // first node: only a hard, theta-dependent initial condition contributes (otherwise S_1 = 0)
        if (has_ic_ && ic_jacobian_) { grad.noalias() += ic_jacobian_(theta).transpose() * p; }
        return grad;
    }

    // outer objective functor over theta, adapting H and its envelope gradient to the core BFGS interface
    struct tracking_objective {
        ts_ls_ode_tracking* solver;
        double operator()(const vector_t& theta) const { return solver->H_(theta); }
        auto gradient() const {
            return [s = solver](const vector_t& theta) { return s->grad_H_(theta); };
        }
    };

    param_engine_handle handle_;
    std::function<matrix_t(const vector_t&)> ic_jacobian_;

    vector_t theta_;                                  // current / estimated parameters
    double lambda_outer_ = -1;
    fit_policy inner_policy_ = fit_policy::adjoint;
    bool policy_override_ = false;
    int outer_max_iter_ = 100;
    double outer_tol_ = 1e-6;
    static constexpr double divergent_outer_cost_ = 1e20;

    // inner-solve memoization
    vector_t theta_cache_;
    bool cache_valid_ = false;
    int inner_failures_ = 0;

    // outer results
    double outer_value_ = 0;
    int outer_n_iter_ = 0;
    bool outer_converged_ = false;
};

}   // namespace internals

// Parametric ODE-penalty descriptor: the theta-parameterized field, the time-integration scheme, an
// initial guess for theta and an optional initial condition. Mirrors the ts_ls_ode descriptor -- the
// stage count and the system dimension are deduced here and erased away -- and additionally carries the
// handle that lets the tracking solver rebind theta.
struct ts_ls_ode_param {
    using solver_t = internals::ts_ls_ode_tracking;
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    struct penalty_packet {
        internals::param_engine_handle handle_;
        vector_t theta0_, ic_;
        bool has_ic_ = false;
        int max_iter_ = 50;
        double tol_ = 1e-10;   // tighter than the forward default: the envelope gradient needs dJ/du ~ 0
       public:
        penalty_packet(internals::param_engine_handle handle, vector_t theta0, int max_iter, double tol) :
            handle_(std::move(handle)), theta0_(std::move(theta0)), max_iter_(max_iter), tol_(tol) { }
        penalty_packet(
          internals::param_engine_handle handle, vector_t theta0, vector_t ic, int max_iter, double tol) :
            handle_(std::move(handle)),
            theta0_(std::move(theta0)),
            ic_(std::move(ic)),
            has_ic_(true),
            max_iter_(max_iter),
            tol_(tol) { }
        // observers (the first five are what ts_ls_ode::discretize consumes)
        internals::any_rk_engine engine() const { return handle_.engine(); }
        const vector_t& ic() const { return ic_; }
        bool has_ic() const { return has_ic_; }
        int max_iter() const { return max_iter_; }
        double tol() const { return tol_; }
        const internals::param_engine_handle& handle() const { return handle_; }
        const vector_t& theta0() const { return theta0_; }
    };
    // Dim is the parametric field's static dimension (fixed-size return -> static, VectorXd -> Dynamic),
    // deduced here and used only to build the typed param_rk_engine; it never escapes this function.
    template <typename Field, int Stages>
    static internals::param_engine_handle make_handle_(
      const Field& field, const ButcherTableau<Stages>& tableau, const vector_t& theta0) {
        constexpr int Dim = ode_rhs_param_dim_v<Field>;
        return internals::make_param_engine_handle<Stages, Dim, Field>(field, RKIntegrator(tableau), theta0);
    }
   public:
    template <typename Field, int Stages>
        requires(is_ode_param_rhs<Field>)
    ts_ls_ode_param(
      const Field& field, const ButcherTableau<Stages>& tableau, const vector_t& theta0, int max_iter = 50,
      double tol = 1e-10) :
        penalty_(make_handle_(field, tableau, theta0), theta0, max_iter, tol) { }
    template <typename Field, int Stages>
        requires(is_ode_param_rhs<Field>)
    ts_ls_ode_param(
      const Field& field, const ButcherTableau<Stages>& tableau, const vector_t& theta0, const vector_t& ic,
      int max_iter = 50, double tol = 1e-10) :
        penalty_(make_handle_(field, tableau, theta0), theta0, ic, max_iter, tol) { }
    const penalty_packet& get() const { return penalty_; }
   private:
    penalty_packet penalty_;
};

}   // namespace fdapde

#endif   // __TS_LS_ODE_TRACKING_SOLVER_H__
