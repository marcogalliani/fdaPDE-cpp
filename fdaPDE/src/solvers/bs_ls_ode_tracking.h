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

#ifndef __BS_LS_ODE_TRACKING_SOLVER_H__
#define __BS_LS_ODE_TRACKING_SOLVER_H__


#include "header_check.h"

namespace fdapde {
namespace internals {

// Tracking inverse solver: estimates the ODE parameters theta of a parameterized prior dynamics
// y' = f(t, y, theta) + u(t) from the same data the forward solver smooths.
//
// Following the tracking estimator (Brunel & Clairon), the outer criterion is the *same* functional as
// the inner one,
//
//   H(theta) = min_u J(u, theta),        theta* = argmin_theta H(theta),
//
// so the whole problem is the joint minimization min_{u, theta} J(u, theta) organized as
// min_theta min_u J(u, theta). It derives from bs_ls_ode and uses the inherited fit() as its inner
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
class bs_ls_ode_tracking : public bs_ls_ode {
   public:
    using bs_ls_ode::fit;   // the inner solver, inherited unchanged
    using Base = bs_ls_ode;

    bs_ls_ode_tracking() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    bs_ls_ode_tracking(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty);
        analyze_data(formula, gf);
    }

    // discretize a parametric penalty: the base takes the (parametric) engine at theta0 -- so a plain
    // forward fit() works immediately -- and the outer solve rebinds theta directly on that inherited engine.
    template <typename Penalty> void discretize(Penalty&& penalty) {
        Base::discretize(penalty);
        theta_ = penalty.theta0();
        fdapde_assert(static_cast<bool>(engine_) && theta_.size() == engine_.n_params());
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
    // Run the outer BFGS on z, theta_k = lower_k + exp(z_k), instead of on theta directly (see
    // theta_log_map). `lower` has one entry per parameter; a NaN entry leaves that component unmapped.
    // The estimate, the gradients and every observer stay in theta.
    void set_parameter_map(const vector_t& lower) {
        fdapde_assert(static_cast<bool>(engine_) && lower.size() == engine_.n_params());
        param_map_.lower = lower;
    }
    void clear_parameter_map() { param_map_.lower.resize(0); }
    const theta_log_map& parameter_map() const { return param_map_; }
    // cap on the length of every accepted BFGS step, in the optimizer's own variables (see
    // step_norm_cap). <= 0 (the default) leaves the step to the line search alone. Ignored by the
    // trust-region outer optimizer, whose radius does the same job with a model behind it.
    void set_outer_step_cap(double cap) { outer_step_cap_ = cap; }
    // which outer optimizer minimizes H_lambda: BFGS with a Wolfe line search (the default) or the
    // trust-region quasi-Newton. They differ in how a step into the divergent region is handled -- the
    // line search shortens it, the trust region rejects it and shrinks the radius.
    // levenberg_marquardt here is Gauss-Newton on the FROZEN-CONTROL residual Jacobian: the gradient
    // it uses is exact (the envelope theorem), the curvature is an upper bound. See tracking_ls_problem.
    enum class outer_method { bfgs, trust_region, levenberg_marquardt };
    void set_outer_method(outer_method m) { outer_method_ = m; }
    outer_method outer_method_policy() const { return outer_method_; }
    // Levenberg-Marquardt only: use the exact reduced (variable-projection) Jacobian (I - P_A)B, the
    // default, or the frozen-control B alone. See tracking_ls_problem.
    void use_exact_jacobian(bool on) { exact_jacobian_ = on; }
    bool exact_jacobian() const { return exact_jacobian_; }
    // d y_0 / d theta (d x n_theta), only meaningful under a hard initial condition that itself depends
    // on theta. Absent (the default) the first node contributes nothing: with no hard IC the initial
    // state is an inner decision variable, so the envelope theorem freezes it too.
    void set_ic_jacobian(std::function<matrix_t(const vector_t&)> ic_jacobian) {
        ic_jacobian_ = std::move(ic_jacobian);
    }
    // Make the hard initial condition itself a function of theta, so the OUTER solve estimates it. `ic`
    // maps theta -> the d-vector that pins node 0 before each inner fit (the inner solve then sees the IC
    // as a fixed constant, NOT a decision variable); `ic_jac` = d(y0)/d(theta) (d x n_theta), contracted
    // against the first-node costate in the envelope gradient. This wires both halves the estimation needs:
    // the forward refresh of the pinned IC (set here) and its gradient term (set_ic_jacobian). Requires a
    // hard-IC discretization -- a free IC makes the initial state an inner variable, which the envelope
    // theorem freezes, so there is nothing for the outer solve to estimate.
    void set_ic_parameterization(std::function<vector_t(const vector_t&)> ic,
                                 std::function<matrix_t(const vector_t&)> ic_jac) {
        fdapde_assert(has_ic_ && static_cast<bool>(ic) && static_cast<bool>(ic_jac));
        ic_from_theta_ = std::move(ic);
        ic_jacobian_ = std::move(ic_jac);
        invalidate_cache_();
    }

    // --- outer solve ---------------------------------------------------
    // minimize H(theta) = min_u J(u, theta) over theta by BFGS on the envelope gradient. On return the
    // inherited forward state (trajectory, control, f(), ...) is the inner fit at the estimated theta.
    const vector_t& solve(double lambda, const vector_t& theta0) {
        fdapde_assert(lambda > 0 && static_cast<bool>(engine_));
        fdapde_assert(theta0.size() == engine_.n_params());
        lambda_outer_ = lambda;
        inner_failures_ = 0;
        best_value_ = std::numeric_limits<double>::infinity();
        best_theta_.resize(0);
        invalidate_cache_();
        tracking_objective problem {this};
        const vector_t z0 = param_map_.z_of(theta0);
        vector_t z_opt;
        if (outer_method_ == outer_method::levenberg_marquardt) {
            LevenbergMarquardt<Dynamic> lm(outer_max_iter_, outer_tol_);
            z_opt = lm.optimize(tracking_ls_problem {this}, z0);
            outer_n_iter_ = lm.n_iter();
            outer_converged_ = lm.converged();
        } else if (outer_method_ == outer_method::trust_region) {
            TrustRegion<Dynamic> optimizer(outer_max_iter_, outer_tol_, outer_step_cap_ > 0 ? outer_step_cap_ : 1.0);
            z_opt = optimizer.optimize(problem, z0);
            outer_n_iter_ = optimizer.n_iter();
            outer_converged_ = optimizer.converged();
        } else {
            BFGS<Dynamic> optimizer(outer_max_iter_, outer_tol_, 1.0);
            // Wolfe line search, as in the inner adjoint solve: the curvature condition keeps the BFGS
            // inverse Hessian positive definite and the Armijo test rejects the divergent-cost surrogate.
            z_opt = optimizer.optimize(problem, z0, WolfeLineSearch(), step_norm_cap {outer_step_cap_});
            outer_n_iter_ = optimizer.n_iter();
            outer_converged_ = optimizer.converged();
        }
        vector_t theta_opt = param_map_.theta_of(z_opt);
        // the optimizer's OWN last iterate, kept before the safeguard below may replace it: the two
        // disagree exactly when the safeguard fires, and a study that reports only the returned point
        // cannot tell a converged run from a rescued one
        theta_last_ = theta_opt;
        outer_value_last_ = theta_opt.allFinite() ? H_(theta_opt) : std::numeric_limits<double>::infinity();
        // SAFEGUARD, as in bs_ls_ode_nls::solve(). Return the best theta actually EVALUATED rather than
        // the optimizer's last iterate: a line search walking into the region where the inner solve
        // diverges gets the sentinel cost and a zeroed gradient, which can leave the BFGS secant update
        // with delta_grad = 0 and hand back a non-finite or uphill iterate -- an artifact of the
        // optimizer, not a statement about the estimator.
        if (!theta_opt.allFinite() || outer_value_last_ > best_value_) {
            fdapde_assert(best_theta_.size() > 0);
            theta_opt = best_theta_;
            safeguarded_ = true;
        } else {
            safeguarded_ = false;
        }
        theta_ = theta_opt;
        // leave the forward state consistent with the estimate
        invalidate_cache_();
        outer_value_ = H_(theta_);   // the inner solve at the estimate, which also refreshes the state
        return theta_;
    }
    const vector_t& solve(double lambda) { return solve(lambda, theta_); }

    // --- observers -----------------------------------------------------
    const vector_t& theta() const { return theta_; }            // parameter estimate
    int n_params() const { return engine_.n_params(); }
    double outer_objective() const { return outer_value_; }     // H(theta*)
    int outer_n_iter() const { return outer_n_iter_; }
    bool outer_converged() const { return outer_converged_; }
    // number of outer evaluations whose inner solve did not meet its tolerance. The envelope gradient is
    // only valid at an inner optimum, so a nonzero count means the outer gradient was inconsistent.
    int n_inner_failures() const { return inner_failures_; }
    // true when the estimate is the best point EVALUATED rather than the optimizer's last iterate; see
    // the SAFEGUARD note in solve()
    bool safeguarded() const { return safeguarded_; }
    // the optimizer's own last iterate and the criterion there, before the safeguard. Reporting a study's
    // failure rate on BOTH this and the returned estimate separates "the optimizer solved it" from "the
    // safeguard found something better among the points a line search happened to try".
    const vector_t& theta_last_iterate() const { return theta_last_; }
    double outer_objective_last_iterate() const { return outer_value_last_; }
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
    // DIAGNOSTIC: 2 J^T r from the least-squares view. The envelope theorem makes this EXACTLY the
    // envelope gradient above (the frozen-control term is the only one that survives), so the two must
    // agree to round-off -- the check on the Jacobian LM uses. Takes the map's variables z (= theta when
    // no parameter map is set).
    /* DIAGNOSTIC: d u^star / d theta itself ((m-1)s*d x n_theta), the quantity the variable projection
    is built on. It needs its OWN test: its contribution to the gradient cancels by the inner optimality
    condition, so ls_gradient_at() is blind to it. The check is a central difference of the fitted
    control against re-solved inner problems, which is what test_varpro does. */
    matrix_t inner_sensitivity_at(double lambda, const vector_t& theta) {
        set_lambda_(lambda);
        inner_solve_(theta);
        if (!Y_.allFinite() || !U_.allFinite()) { return matrix_t(); }
        const auto& idx = obs_idx_();
        const int n_obs = static_cast<int>(idx.size()), nu = n_ctrl_(), p = engine_.n_params();
        matrix_t S0 = matrix_t::Zero(d_, p);
        if (has_ic_ && ic_jacobian_) { S0 = ic_jacobian_(theta); }
        const auto S = param_sensitivity_(Y_, &U_, S0, m_);
        matrix_t B_data = matrix_t::Zero(n_obs, p);
        for (int i = 0; i < n_obs; ++i) {
            const auto [t, v] = idx[i];
            if (t < static_cast<int>(S.size())) { B_data.row(i) = S[t].row(v); }
        }
        const matrix_t C = control_sensitivity_(Y_, U_, idx);
        const vector_t w = penalty_weight_();
        Eigen::SparseMatrix<double> M(nu, nu);
        M.reserve(Eigen::VectorXi::Constant(nu, 1));
        for (int j = 0; j < nu; ++j) { M.insert(j, j) = w[j]; }
        M.makeCompressed();
        return inner_sensitivity_(C, M, B_data);
    }
    // the fitted control at (lambda, theta), flattened as inner_sensitivity_at differentiates it
    vector_t control_at(double lambda, const vector_t& theta) {
        set_lambda_(lambda);
        inner_solve_(theta);
        return flat_control_();
    }

    vector_t ls_gradient_at(double lambda, const vector_t& z) {
        set_lambda_(lambda);
        tracking_ls_problem p {this};
        return 2.0 * (p.jacobian(z).transpose() * p.residuals(z));
    }

   private:
    // run the inner solve at theta, memoized: BFGS asks for the objective and the gradient at the same
    // point, and one inner solve serves both.
    void inner_solve_(const vector_t& theta) {
        if (cache_valid_ && theta_cache_.size() == theta.size() && theta_cache_ == theta) { return; }
        engine_.set_theta(theta);   // rebind the inherited engine to the new parameters, in place
        if (ic_from_theta_) { y0_ = ic_from_theta_(theta); }   // refresh the theta-dependent hard IC (pinned node 0)
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
        if (!std::isfinite(v) || !Y_.allFinite()) { v = divergent_outer_cost_; }
        // running best, for the safeguard in solve()
        if (v < best_value_ && theta.allFinite()) {
            best_value_ = v;
            best_theta_ = theta;
        }
        return v;
    }

    // envelope gradient: a single backward adjoint sweep at frozen control.
    vector_t grad_H_(const vector_t& theta) {
        inner_solve_(theta);
        const int n_theta = engine_.n_params();
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
            const matrix_t ut = stage_block_(U_, t);
            // one stage solve yields both the parameter Jacobian Theta_t = d y_{t+1}/d theta and the flow
            // Jacobian Flow_t = d y_{t+1}/d y_t on the forced dynamics; p is the costate p_{t+1}. The envelope
            // gradient contracts Theta_t against p, the costate propagates by Flow_t^T (pure linear algebra --
            // no second stage solve, cf. the previous param_jacobian + adjoint_step pair).
            auto s = engine_.step_with_flow_param_jacobians(time_(t), yc, dt_(t), ut);
            grad.noalias() += s.param.transpose() * p;
            p = source(t) + s.flow.transpose() * p;
        }
        // first node: only a hard, theta-dependent initial condition contributes (otherwise S_1 = 0)
        if (has_ic_ && ic_jacobian_) { grad.noalias() += ic_jacobian_(theta).transpose() * p; }
        return grad;
    }

    // outer objective functor over the optimizer's variables z (theta itself under the identity map),
    // adapting H and its envelope gradient to the core BFGS interface
    /* The penalized criterion H_lambda as residuals and a Jacobian, for Levenberg-Marquardt.

    H_lambda(theta) = ||R(u^star(theta), theta)||^2 with R = [ y - y_obs ; sqrt(lambda W) u ], so a
    least-squares view exists -- but the reduced Jacobian is a TOTAL derivative,

        dR/dtheta = B + A (d u^star / d theta) ,     A = dR/du ,  B = partial R / partial theta ,

    and d u^star/d theta is what the envelope theorem lets the gradient avoid. It does NOT let the
    curvature avoid it. Substituting the implicit-function solution (see inner_sensitivity_),

        dR/dtheta = ( I - P_A ) B ,     P_A = A (A^T A)^{-1} A^T ,

    the frozen-control Jacobian PROJECTED off the column space of A -- Golub & Pereyra's variable
    projection. Three things follow, and the first two are why the earlier frozen-control version was
    both correct in its gradient and bad in its steps:

      GRADIENT unchanged. P_A R = 0 IS the inner optimality condition, so B^T (I - P_A) R = B^T R. The
      ls_gradient_at() check still has to reproduce the envelope gradient exactly -- and now it tests more,
      since a poorly converged inner solve leaves A^T R != 0 and the two drift apart.

      CURVATURE corrected downwards. B^T(I - P_A)B <= B^T B, the gap being ||P_A B||^2. Using B alone
      over-states curvature, so the damped steps come out too short: measured on SB, LM on the
      frozen-control model burned its whole outer budget and converged on none of 20 replicas, while the
      trust region reached the lower criterion on 16 of them.

      COVARIANCE. sigma^2 (J^T J)^{-1} with the projected J is the estimator's actual asymptotic
      covariance; the frozen-control model understates it.

    GAUSS-NEWTON ON THE INNER PROBLEM TOO. The implicit-function solution uses A^T A for the inner
    Hessian, dropping sum_i R_i d^2 R_i/du^2. So d u^star/d theta is exact only as the residuals vanish.
    Measured against central differences of re-solved inner problems (jacobian_checks.cpp): ~2e-5 with
    clean data at the truth, 14-20% at sigma = 5 with theta 20% out, and INDEPENDENT of the step size, so
    it is the approximation and not finite-difference noise. For a damped method that is acceptable --
    the rho test absorbs model error, and a 15%-wrong curvature beats the frozen-control alternative,
    which is wrong by ||P_A B||^2 -- but it is worth knowing before quoting (J^T J)^{-1} as a covariance.

    Set use_exact_jacobian(false) to fall back to B alone -- kept for comparison, not recommended.
    */
    struct tracking_ls_problem {
        bs_ls_ode_tracking* s;
        // R = [ data residuals ; sqrt(lambda dt_t b_i) u_{t,i} ], so ||R||^2 == H_lambda exactly
        Eigen::Matrix<double, Dynamic, 1> residuals(const vector_t& z) const {
            const vector_t theta = s->param_map_.theta_of(z);
            s->inner_solve_(theta);
            const auto& idx = s->obs_idx_();
            const int n_obs = static_cast<int>(idx.size()), nu = s->n_ctrl_();
            Eigen::Matrix<double, Dynamic, 1> r(n_obs + nu);
            if (!s->Y_.allFinite() || !s->U_.allFinite() || !std::isfinite(s->objective())) {
                return r.setConstant(std::numeric_limits<double>::quiet_NaN());
            }
            for (int i = 0; i < n_obs; ++i) {
                const auto [t, v] = idx[i];
                r[i] = s->Y_(t, v) - s->y_obs_(t, v);
            }
            r.tail(nu) = s->penalty_weight_().cwiseSqrt().cwiseProduct(s->flat_control_());
            return r;
        }
        Eigen::Matrix<double, Dynamic, Dynamic> jacobian(const vector_t& z) const {
            const vector_t theta = s->param_map_.theta_of(z);
            s->inner_solve_(theta);
            const auto& idx = s->obs_idx_();
            const int n_obs = static_cast<int>(idx.size()), nu = s->n_ctrl_(), p = s->engine_.n_params();
            Eigen::Matrix<double, Dynamic, Dynamic> J =
              Eigen::Matrix<double, Dynamic, Dynamic>::Zero(n_obs + nu, p);
            if (!s->Y_.allFinite() || !s->U_.allFinite()) { return J; }
            // B_data = d y / d theta at FROZEN control. Only a hard, theta-dependent initial condition
            // moves y_0; with a free IC the initial state is an inner variable, frozen like the control.
            matrix_t S0 = matrix_t::Zero(s->d_, p);
            if (s->has_ic_ && s->ic_jacobian_) { S0 = s->ic_jacobian_(theta); }
            const auto S = s->param_sensitivity_(s->Y_, &s->U_, S0, s->m_);
            matrix_t B_data = matrix_t::Zero(n_obs, p);
            for (int i = 0; i < n_obs; ++i) {
                const auto [t, v] = idx[i];
                if (t < static_cast<int>(S.size())) { B_data.row(i) = S[t].row(v); }
            }
            J.topRows(n_obs) = B_data;
            if (s->exact_jacobian_) {
                const matrix_t C = s->control_sensitivity_(s->Y_, s->U_, idx);
                const vector_t w = s->penalty_weight_();
                Eigen::SparseMatrix<double> M(nu, nu);   // M = P^T P, diagonal for the L2 penalty
                M.reserve(Eigen::VectorXi::Constant(nu, 1));
                for (int j = 0; j < nu; ++j) { M.insert(j, j) = w[j]; }
                M.makeCompressed();
                const matrix_t du = s->inner_sensitivity_(C, M, B_data);
                if (du.size() != 0) {
                    J.topRows(n_obs) += C * du;                              // data block
                    J.bottomRows(nu) = w.cwiseSqrt().asDiagonal() * du;      // penalty block
                }
            }
            if (s->param_map_.active()) {
                const vector_t d = s->param_map_.chain(theta, vector_t::Ones(p));
                for (int j = 0; j < p; ++j) { J.col(j) *= d[j]; }
            }
            return J;
        }
    };
    // the per-block L2 penalty weight lambda*dt_t*b_i, as one vector over the (m-1)s*d control unknowns,
    // and the fitted control flattened to match. ||sqrt(w) u||^2 is then exactly the penalty term of H.
    vector_t penalty_weight_() const {
        const vector_t b = quad_weights_();
        vector_t w(n_ctrl_());
        for (int t = 0; t < m_ - 1; ++t) {
            for (int i = 0; i < s_; ++i) {
                w.segment((t * s_ + i) * d_, d_).setConstant(lambda_outer_ * dt_(t) * b[i]);
            }
        }
        return w;
    }
    vector_t flat_control_() const {
        vector_t u(n_ctrl_());
        for (int t = 0; t < m_ - 1; ++t) { u.segment(t * sd_(), sd_()) = U_.row(t).transpose(); }
        return u;
    }
    // observed (node, component) pairs, cached: the residual layout must be stable across iterations
    const std::vector<std::pair<int, int>>& obs_idx_() {
        if (obs_idx_cache_.empty() && n_obs_ > 0) { obs_idx_cache_ = observed_entries_(); }
        return obs_idx_cache_;
    }
    std::vector<std::pair<int, int>> obs_idx_cache_;

    struct tracking_objective {
        bs_ls_ode_tracking* solver;
        double operator()(const vector_t& z) const { return solver->H_(solver->param_map_.theta_of(z)); }
        auto gradient() const {
            return [s = solver](const vector_t& z) {
                const vector_t theta = s->param_map_.theta_of(z);
                vector_t g = s->grad_H_(theta);
                // GUARD. Outside the region where the field is defined the sweep returns non-finite
                // entries. BFGS's stopping test reads grad.norm() > tol, which is FALSE for NaN, so an
                // unguarded NaN ends the outer loop and reports the garbage point as converged. Zeroing it
                // stops the loop too, but on a point the safeguard in solve() can then discard.
                if (!g.allFinite()) { g.setZero(); }
                return s->param_map_.chain(theta, g);
            };
        }
    };

    std::function<matrix_t(const vector_t&)> ic_jacobian_;
    std::function<vector_t(const vector_t&)> ic_from_theta_;   // theta-dependent hard IC, refreshed per inner solve

    vector_t theta_;                                  // current / estimated parameters
    double lambda_outer_ = -1;
    fit_policy inner_policy_ = fit_policy::adjoint;
    bool policy_override_ = false;
    int outer_max_iter_ = 100;
    double outer_tol_ = 1e-6;
    static constexpr double divergent_outer_cost_ = 1e20;
    theta_log_map param_map_;       // identity unless set_parameter_map() was called
    bool exact_jacobian_ = true;    // LM: variable-projection Jacobian rather than the frozen-control one
    double outer_step_cap_ = 0;     // <= 0: the step length is left to the line search

    // best evaluated point, the fallback solve() returns when the optimizer's last iterate is not usable
    vector_t best_theta_;
    double best_value_ = std::numeric_limits<double>::infinity();
    bool safeguarded_ = false;

    // inner-solve memoization
    vector_t theta_cache_;
    bool cache_valid_ = false;
    int inner_failures_ = 0;

    // outer results
    vector_t theta_last_;            // the optimizer's last iterate, before the safeguard
    double outer_value_last_ = std::numeric_limits<double>::infinity();
    outer_method outer_method_ = outer_method::bfgs;
    double outer_value_ = 0;
    int outer_n_iter_ = 0;
    bool outer_converged_ = false;
};

}   // namespace internals

// Parametric ODE-penalty descriptor: the theta-parameterized field, the time-integration scheme, an
// initial guess for theta and an optional initial condition. Mirrors the bs_ls_ode descriptor -- the stage
// count and the system dimension are deduced here and erased away -- the only difference being that the
// erased engine wraps a parametric field (theta bound to theta0), so the tracking solver rebinds theta
// directly on the inherited any_controlled_ode_solver.
struct bs_ls_ode_param {
    using solver_t = internals::bs_ls_ode_tracking;
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    struct penalty_packet {
        any_controlled_ode_solver engine_;
        std::vector<std::string> names_;   // component names of the rhs form; empty for a plain functor
        vector_t theta0_, ic_;
        bool has_ic_ = false;
        int degree_ = 1;   // degree of the declared trajectory space == stage count of the scheme
        const BsSpace<Triangulation<1, 1>>* space_ = nullptr;   // the trajectory space, read by discretize
        int max_iter_ = 500;
        double tol_ = 1e-10;   // tighter than the forward default: the envelope gradient needs dJ/du ~ 0
       public:
        penalty_packet(
          any_controlled_ode_solver engine, int degree, const BsSpace<Triangulation<1, 1>>* space, vector_t theta0,
          int max_iter, double tol) :
            engine_(std::move(engine)), theta0_(std::move(theta0)), degree_(degree), space_(space),
            max_iter_(max_iter), tol_(tol) { }
        penalty_packet(
          any_controlled_ode_solver engine, int degree, const BsSpace<Triangulation<1, 1>>* space, vector_t theta0,
          vector_t ic, int max_iter, double tol) :
            engine_(std::move(engine)),
            theta0_(std::move(theta0)),
            ic_(std::move(ic)),
            has_ic_(true),
            degree_(degree),
            space_(space),
            max_iter_(max_iter),
            tol_(tol) { }
        // observers (the first seven are what bs_ls_ode::discretize consumes)
        int degree() const { return degree_; }
        const BsSpace<Triangulation<1, 1>>* space() const { return space_; }
        const any_controlled_ode_solver& engine() const { return engine_; }
        const std::vector<std::string>& component_names() const { return names_; }
        void set_component_names(std::vector<std::string> names) { names_ = std::move(names); }
        const vector_t& ic() const { return ic_; }
        bool has_ic() const { return has_ic_; }
        int max_iter() const { return max_iter_; }
        double tol() const { return tol_; }
        const vector_t& theta0() const { return theta0_; }
    };
    // build the erased control-aware engine over the parametric field with theta0 bound. Dim is the field's
    // static dimension (fixed-size return -> static, VectorXd -> Dynamic), deduced here and used only to
    // build the typed controlled_ode_solver; it never escapes this function.
    template <typename Field, int Stages>
    static any_controlled_ode_solver engine_from_tableau_(
      const Field& field, const ButcherTableau<Stages>& tableau, const vector_t& theta0) {
        constexpr int Dim = ode_rhs_dim_v<Field>;
        ode_rhs_field<Dim, Field> f(field);
        f.set_theta(theta0);
        return any_controlled_ode_solver(
          controlled_ode_solver<Stages, Dim, Field>(std::move(f), RKIntegrator(tableau)));
    }
    // the trajectory space's degree selects the p-stage Gauss scheme; see bs_ls_ode::engine_from_degree_
    template <typename Field>
    static any_controlled_ode_solver
    engine_from_degree_(const Field& field, int degree, const vector_t& theta0) {
        switch (degree) {
        case 1: return engine_from_tableau_(field, ode_schemes::implicit_midpoint(), theta0);
        case 2: return engine_from_tableau_(field, ode_schemes::gauss_legendre_2(), theta0);
        case 3: return engine_from_tableau_(field, ode_schemes::gauss_legendre_3(), theta0);
        case 4: return engine_from_tableau_(field, ode_schemes::gauss_legendre_4(), theta0);
        }
        fdapde_assert(
          false && "trajectory space degree must be 1, 2, 3 or 4 (the Gauss-Legendre schemes available)");
        return any_controlled_ode_solver();
    }
   public:
    template <typename Field>
        requires(is_parameterized_ode_rhs<Field> && !is_ode_system_v<Field>)
    bs_ls_ode_param(
      const Field& field, const BsSpace<Triangulation<1, 1>>& Vh, const vector_t& theta0, int max_iter = 500,
      double tol = 1e-10) :
        penalty_(
          engine_from_degree_(field, Vh.order(), theta0), Vh.order(), std::addressof(Vh), theta0, max_iter,
          tol) { }
    template <typename Field>
        requires(is_parameterized_ode_rhs<Field> && !is_ode_system_v<Field>)
    bs_ls_ode_param(
      const Field& field, const BsSpace<Triangulation<1, 1>>& Vh, const vector_t& theta0, const vector_t& ic,
      int max_iter = 500, double tol = 1e-10) :
        penalty_(
          engine_from_degree_(field, Vh.order(), theta0), Vh.order(), std::addressof(Vh), theta0, ic, max_iter,
          tol) { }
    // from a system in strong form: the space comes from its unknowns, theta0 follows with_parameters' order
    template <typename... Eqs>
    bs_ls_ode_param(const ode_system<Eqs...>& sys, const vector_t& theta0, int max_iter = 500, double tol = 1e-10) :
        penalty_(
          engine_from_degree_(sys, sys.function_space().order(), theta0), sys.function_space().order(),
          std::addressof(sys.function_space()), theta0, max_iter, tol) {
        fdapde_assert(theta0.size() == sys.n_params() && "theta0 must have one entry per declared parameter");
        penalty_.set_component_names(sys.component_names());
    }
    template <typename... Eqs>
    bs_ls_ode_param(
      const ode_system<Eqs...>& sys, const vector_t& theta0, const vector_t& ic, int max_iter = 500,
      double tol = 1e-10) :
        penalty_(
          engine_from_degree_(sys, sys.function_space().order(), theta0), sys.function_space().order(),
          std::addressof(sys.function_space()), theta0, ic, max_iter, tol) {
        fdapde_assert(theta0.size() == sys.n_params() && "theta0 must have one entry per declared parameter");
        penalty_.set_component_names(sys.component_names());
    }
    const penalty_packet& get() const { return penalty_; }
   private:
    penalty_packet penalty_;
};

}   // namespace fdapde

#endif   // __BS_LS_ODE_TRACKING_SOLVER_H__
