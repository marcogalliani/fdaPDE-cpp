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

#ifndef __TS_LS_ODE_NLS_SOLVER_H__
#define __TS_LS_ODE_NLS_SOLVER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// Nonlinear least squares (single shooting) inverse solver: estimates the ODE parameters theta of a
// parameterized dynamics y' = f(t, y, theta) by fitting the EXACT solution of that ODE to the data,
//
//   S(theta, y0) = sum_{t,v} m_{t,v} ( y_{t,v}(theta, y0) - y^obs_{t,v} )^2,
//   Y(theta, y0) = the discrete trajectory obtained by integrating f from y0 over the time grid,
//
// minimized over theta (and, unless the initial state is pinned, over y0 as well).
//
// WHY IT EXISTS. It is the reference estimator the tracking solver (ts_ls_ode_tracking) is to be measured
// against. The tracking criterion H_lambda(theta) = min_u J(u, theta) admits a control u that lets the
// trajectory leave the model manifold, and the claim motivating it (Ramsay & Hooker's generalized
// profiling) is precisely that H_lambda is a SMOOTHED version of S, approaching it as lambda -> infinity.
// Testing that claim requires S itself, computed on the SAME model, the SAME time grid and the SAME
// Butcher tableau -- otherwise a comparison of estimators is partly a comparison of discretizations.
// Hence this solver consumes the same ts_ls_ode_param descriptor and the same erased engine as the
// tracking solver, and derives from the same ts_ls_ode base so data ingestion, the observation mask and
// the result observers are literally the same code.
//
// RELATION TO THE TRACKING SOLVER. NLS is the u == 0 corner of the same machinery: there is no inner
// problem, no penalty and no lambda -- the state is not a decision variable but a function of theta.
// The gradient is correspondingly NOT an envelope gradient but the exact discrete adjoint of the
// shooting map, assembled by the same backward sweep at zero control,
//
//   p = s_{m-1};  for t = m-2..0:  grad += Theta_t^T p;  p = s_t + Flow_t^T p,
//
// with s_t = 2 m_t (y_t - y^obs_t), and Theta_t = d y_{t+1}/d theta, Flow_t = d y_{t+1}/d y_t supplied by
// the engine's step_with_flow_param_jacobians on the UNFORCED dynamics (u = 0). The costate left at node
// 0 is dS/dy_0, which is exactly what the initial-state block of the gradient needs -- so estimating y0
// jointly with theta costs nothing extra. The identical sweep with a nonzero control and the envelope
// argument gives the tracking gradient: the two estimators differ in their criterion, not in their
// sensitivity machinery.
//
// The outer minimization is BFGS with a Wolfe line search from the core optimization module, as in the
// tracking solver's outer loop -- the difference in behaviour between the two estimators is then a
// property of the criteria, not of the optimizer.
class ts_ls_ode_nls : public ts_ls_ode {
   public:
    using Base = ts_ls_ode;
    /* Line search for the outer BFGS
        - backtracking: Armijo backtracking from a fixed initial step, halving without a floor.
        - wolfe:        the weak Wolfe (Armijo + curvature) bisection used by the tracking solver.
       They are NOT interchangeable here, and the reason is worth stating: at the first BFGS iteration the
       inverse Hessian is the identity, so the search direction is the raw gradient, whose norm on a
       shooting criterion is set by the sensitivity of a whole trajectory to theta -- routinely in the
       hundreds. The Wolfe implementation bisects at most 10 times from alpha = 1, i.e. it cannot go below
       ~1e-3 of that direction, which on FitzHugh-Nagumo overshoots the descent region by an order of
       magnitude and can spend the run recovering. Backtracking has no such floor, so it is the default.
       Wolfe is kept because it is what the tracking solver runs, and comparing the two estimators under
       the *same* line search is sometimes the point. */
    enum class line_search { backtracking, wolfe };

    ts_ls_ode_nls() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    ts_ls_ode_nls(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty);
        analyze_data(formula, gf);
    }

    // discretize a parametric descriptor (ts_ls_ode_param): the base stores the erased engine at theta0
    // and the optional hard initial condition, this solve rebinds theta on that engine per iterate. The
    // control machinery the base sets up is simply never exercised: NLS integrates at u = 0.
    template <typename Penalty> void discretize(Penalty&& penalty) {
        Base::discretize(penalty);
        theta_ = penalty.theta0();
        fdapde_assert(static_cast<bool>(engine_) && theta_.size() == engine_.n_params());
        estimate_ic_ = !has_ic_;   // a pinned initial state is data, a free one is an unknown (see below)
        invalidate_cache_();
    }

    // ingest the data. The base overloads do the work; NLS additionally needs a starting value for the
    // initial state, which is only definable once the observations are known (the shooting map is
    // undefined without a y_0, and every entry point -- solve(), gradient_at(), decision_vector() -- may
    // be called before solve()).
    void analyze_data(const vector_t& time_nodes, const matrix_t& y_obs) {
        Base::analyze_data(time_nodes, y_obs);
        init_ic_();
    }
    template <typename GeoFrame> void analyze_data(const std::string& formula, const GeoFrame& gf) {
        Base::analyze_data(formula, gf);
        init_ic_();
    }
    template <typename GeoFrame, typename WeightMatrix>
    void analyze_data(const std::string& formula, const GeoFrame& gf, const WeightMatrix&) {
        analyze_data(formula, gf);
    }

    // --- configuration -------------------------------------------------
    void set_options(int max_iter, double tol) {
        fdapde_assert(max_iter > 0 && tol > 0);
        outer_max_iter_ = max_iter;
        outer_tol_ = tol;
    }
    void set_line_search(line_search ls) { line_search_ = ls; }
    line_search line_search_policy() const { return line_search_; }
    // Estimate the initial state jointly with theta (single shooting's standard nuisance parameter).
    // Defaults to ON when the discretization carries no hard initial condition -- with a free y0 the
    // shooting map is undefined until something fixes it, and leaving it at an arbitrary value would
    // bias theta -- and OFF under a hard IC, where y0 is given. Switching it off without a hard IC pins
    // y0 at the start value (set_ic_start, else the observation-interpolated first node).
    void estimate_ic(bool on) {
        fdapde_assert(!(on && has_ic_ && ic_from_theta_));
        estimate_ic_ = on;
        invalidate_cache_();
    }
    bool estimates_ic() const { return estimate_ic_; }
    // starting value of the estimated initial state (default: the observations interpolated at t_0)
    void set_ic_start(const vector_t& y0) {
        y0_ = y0;
        invalidate_cache_();
    }
    // d y_0 / d theta (d x n_theta), for an initial state that is itself a function of theta
    void set_ic_jacobian(std::function<matrix_t(const vector_t&)> ic_jacobian) {
        ic_jacobian_ = std::move(ic_jacobian);
    }
    // Make the initial condition a function of theta, as in the tracking solver: `ic` maps theta -> the
    // d-vector the shooting starts from and `ic_jac` = d(y0)/d(theta) is contracted against the node-0
    // costate. Turns the initial state from a free nuisance parameter into a deterministic function of
    // the estimated parameters, so it is NOT estimated separately (estimate_ic is turned off).
    void set_ic_parameterization(
      std::function<vector_t(const vector_t&)> ic, std::function<matrix_t(const vector_t&)> ic_jac) {
        fdapde_assert(static_cast<bool>(ic) && static_cast<bool>(ic_jac));
        ic_from_theta_ = std::move(ic);
        ic_jacobian_ = std::move(ic_jac);
        estimate_ic_ = false;
        invalidate_cache_();
    }

    // --- solve ---------------------------------------------------------
    // minimize S over the decision vector z (theta, and y0 when estimated) by BFGS on the exact discrete
    // adjoint gradient. On return the inherited forward state (trajectory(), f(), misfit(), rss(), ...)
    // is the shooting solution at the estimate.
    const vector_t& solve(const vector_t& theta0) {
        fdapde_assert(static_cast<bool>(engine_) && theta0.size() == engine_.n_params());
        fdapde_assert(m_ > 0 && d_ > 0);   // analyze_data must have run
        theta_ = theta0;
        if (y0_.size() != d_) { init_ic_(); }   // (analyze_data already did this; a re-discretize may not have)
        invalidate_cache_();
        best_value_ = std::numeric_limits<double>::infinity();
        best_z_.resize(0);
        n_divergent_ = 0;
        nls_objective problem {this};
        BFGS<Dynamic> optimizer(outer_max_iter_, outer_tol_, 1.0);
        // the line search is what makes the divergent-cost surrogate work: an unstable theta yields a huge
        // but finite S, which the Armijo test rejects, so the step is cut back into the stable region.
        vector_t z_opt = (line_search_ == line_search::wolfe) ?
                           optimizer.optimize(problem, decision_vector(), WolfeLineSearch()) :
                           optimizer.optimize(problem, decision_vector(), BacktrackingLineSearch());
        // SAFEGUARD. Return the best decision vector actually EVALUATED, not blindly the optimizer's last
        // iterate. On a criterion as unforgiving as the shooting one, a line search that walks into the
        // region where the forward integration diverges gets the sentinel cost and (by construction) a
        // zero gradient, which can leave the BFGS secant update with delta_grad = 0 and hand back a
        // non-finite iterate -- an artifact of the optimizer, not a statement about the estimator. The
        // best evaluated point is never worse than the last one, and is always finite when any evaluation
        // was.
        if (!z_opt.allFinite() || S_(z_opt) > best_value_) {
            fdapde_assert(best_z_.size() > 0);
            z_opt = best_z_;
            safeguarded_ = true;
        } else {
            safeguarded_ = false;
        }
        theta_ = theta_of_(z_opt);
        if (estimate_ic_) { y0_ = z_opt.tail(d_); }
        outer_value_ = S_(z_opt);
        outer_n_iter_ = optimizer.n_iter();
        outer_converged_ = (outer_n_iter_ < outer_max_iter_);
        write_state_(z_opt);   // leave the forward state consistent with the estimate
        return theta_;
    }
    const vector_t& solve() { return solve(theta_); }

    // --- observers -----------------------------------------------------
    // named as in ts_ls_ode_tracking, so a driver reports both estimators through the same calls. There
    // is no inner solve here, so "outer" reads simply as "the estimation problem".
    const vector_t& theta() const { return theta_; }              // parameter estimate
    const vector_t& initial_state() const { return y0_; }         // the (possibly estimated) y_0
    int n_params() const { return engine_.n_params(); }
    double outer_objective() const { return outer_value_; }       // S at the estimate
    int outer_n_iter() const { return outer_n_iter_; }
    // NOTE, as for the tracking solver: this is `n_iter < max_iter`, so an outer solve that exits
    // immediately -- e.g. because the very first forward integration diverged and returned the sentinel
    // cost -- is also reported as converged, with the estimate left at the starting guess. Classify runs
    // by a fit statistic (rss(), or the trajectory error against a known truth), not by this flag alone.
    bool outer_converged() const { return outer_converged_; }
    // number of forward integrations that did not reach the end of the horizon (a diagnostic: a large
    // count means the search spent its budget in the unstable region of parameter space)
    int n_divergent_evals() const { return n_divergent_; }
    // true when the returned estimate is the best point EVALUATED rather than the optimizer's last
    // iterate. Routine, not a failure: the line search evaluates points the iterate sequence does not
    // keep. See the SAFEGUARD note in solve().
    bool safeguarded() const { return safeguarded_; }

    // the decision vector layout: z = [theta] or z = [theta; y_0] when the initial state is estimated
    int n_unknowns() const { return n_params() + (estimate_ic_ ? d_ : 0); }
    vector_t decision_vector() const {
        fdapde_assert(!estimate_ic_ || y0_.size() == d_);
        vector_t z(n_unknowns());
        z.head(theta_.size()) = theta_;
        if (estimate_ic_) { z.tail(d_) = y0_; }
        return z;
    }
    // S and its gradient at an explicit decision vector, exposed for diagnostics (objective surfaces) and
    // for the finite-difference cross-check of the adjoint gradient. Memoization collapses an
    // objective/gradient pair at the same z onto a single forward integration.
    double objective_at(const vector_t& z) { return S_(z); }
    vector_t gradient_at(const vector_t& z) { return grad_S_(z); }
    // the shooting trajectory (m x d) at an explicit theta and the current initial state, without
    // touching the solver's own state
    matrix_t trajectory_at(const vector_t& theta) {
        vector_t z = decision_vector();
        z.head(theta.size()) = theta;
        return forward_(z);
    }

   private:
    // the base's forward smoother and its state-box constraints are not part of this solver's problem:
    // NLS has no control to penalize and no lambda to fit at. Hiding them keeps the API honest (the
    // machinery itself is still there, used through the engine at u = 0).
    using Base::fit;
    using Base::set_state_bounds;
    using Base::clear_state_bounds;
    using Base::has_state_bounds;

    vector_t theta_of_(const vector_t& z) const { return z.head(engine_.n_params()); }
    vector_t y0_of_(const vector_t& z, const vector_t& theta) const {
        if (estimate_ic_) { return z.tail(d_); }
        if (ic_from_theta_) { return ic_from_theta_(theta); }
        return y0_;
    }
    // starting value of the initial state: the observations interpolated at t_0 (the base's own smoothing
    // start), unless a hard initial condition already pinned it
    void init_ic_() {
        if (!has_ic_ && m_ > 0 && d_ > 0) { y0_ = initial_guess_().row(0).transpose(); }
        invalidate_cache_();
        return;
    }
    void invalidate_cache_() {
        fwd_valid_z_ = false;
        z_cache_.resize(0);
        return;
    }

    // forward integration of the prior dynamics at the parameters carried by z, memoized: BFGS and the
    // line search ask for the objective and the gradient at the same point, and one integration serves
    // both (the gradient's backward sweep re-solves the stages per interval anyway, since it needs the
    // Jacobians the plain step does not produce).
    const matrix_t& forward_(const vector_t& z) {
        if (fwd_valid_z_ && z_cache_.size() == z.size() && z_cache_ == z) { return Yz_; }
        vector_t theta = theta_of_(z);
        engine_.set_theta(theta);
        // u == 0 on every interval: the NLS trajectory is the *unforced* solution of the ODE
        Yz_ = engine_.solve(time_, y0_of_(z, theta), matrix_t::Zero(m_ - 1, d_));
        // how far the integration stayed finite (== m_ when it completed): everything below works on this
        // prefix, so an unstable parameter still yields a usable criterion and gradient
        n_finite_ = m_;
        for (int t = 0; t < m_; ++t) {
            if (!Yz_.row(t).allFinite()) {
                n_finite_ = t;
                break;
            }
        }
        if (n_finite_ < m_) { ++n_divergent_; }
        z_cache_ = z;
        fwd_valid_z_ = true;
        return Yz_;
    }
    // S(z): the shooting sum of squares.
    //
    // DIVERGENT PARAMETERS. Nothing constrains theta to the region where the dynamics are stable, and on a
    // stiff or excitable system a line search reaches that region routinely -- FitzHugh-Nagumo blows an
    // explicit scheme up well inside the plausible parameter range. Returning a flat sentinel there costs
    // the solver its way back: the gradient of a constant is zero, and a zero gradient leaves the BFGS
    // secant update with delta_grad = 0, i.e. a NaN inverse Hessian and a dead run.
    //
    // So a blown-up integration is scored on WHAT IT MANAGED TO INTEGRATE: the residuals over the finite
    // prefix, plus a sentinel proportional to the fraction of the horizon that was lost,
    //
    //     S = sum_{t < k} m_t (y_t - y^obs_t)^2 + divergent_cost_ * (m - k)/m,     k = finite prefix length.
    //
    // Two properties matter. (i) Any divergent point costs at least divergent_cost_/m, which dominates any
    // attainable finite value, so the Armijo test still rejects every step out of the stable region.
    // (ii) Among divergent points, integrating FURTHER is better, and the prefix residuals give a genuine
    // descent direction back towards the data -- so a solve *started* at an unstable guess can recover
    // instead of dying at the first evaluation.
    double S_(const vector_t& z) {
        const matrix_t& Y = forward_(z);
        double s = 0;
        for (int t = 0; t < n_finite_; ++t) {
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) {
                    double r = Y(t, v) - y_obs_(t, v);
                    s += r * r;
                }
            }
        }
        if (!std::isfinite(s)) { return divergent_cost_; }
        if (n_finite_ < m_) { return s + divergent_cost_ * double(m_ - n_finite_) / m_; }
        if (s < best_value_ && z.allFinite()) {   // the safeguard's running best (see solve())
            best_value_ = s;
            best_z_ = z;
        }
        return s;
    }
    // dS/dz by the exact discrete adjoint of the shooting map: one backward sweep over the intervals.
    vector_t grad_S_(const vector_t& z) {
        const matrix_t& Y = forward_(z);
        const int n_theta = engine_.n_params();
        vector_t grad = vector_t::Zero(z.size());
        // sweep the finite prefix only; nothing was integrated (a non-finite z) -> zero gradient
        const int k = n_finite_;
        if (k < 2) { return grad; }
        auto source = [&](int t) {   // dS/dy_t at frozen trajectory: the data residual at the node
            vector_t s = vector_t::Zero(d_);
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { s(v) = 2.0 * (Y(t, v) - y_obs_(t, v)); }
            }
            return s;
        };
        const vector_t no_control = vector_t::Zero(d_);
        vector_t p = source(k - 1);
        for (int t = k - 2; t >= 0; --t) {
            vector_t yc = Y.row(t).transpose();
            // one stage solve yields both Theta_t = d y_{t+1}/d theta and Flow_t = d y_{t+1}/d y_t on the
            // unforced dynamics; p is the costate p_{t+1}
            auto s = engine_.step_with_flow_param_jacobians(time_(t), yc, dt_(t), no_control);
            grad.head(n_theta).noalias() += s.param.transpose() * p;
            p = source(t) + s.flow.transpose() * p;
        }
        // node 0: p is now dS/dy_0. It is the gradient block of an estimated initial state, or -- for an
        // initial state parameterized by theta -- feeds the parameter gradient through d(y0)/d(theta).
        if (estimate_ic_) {
            grad.tail(d_) = p;
        } else if (ic_jacobian_) {
            grad.head(n_theta).noalias() += ic_jacobian_(theta_of_(z)).transpose() * p;
        }
        if (!grad.allFinite()) { return vector_t::Zero(z.size()); }
        return grad;
    }
    // fill the inherited result members from the solution at z, so trajectory()/f()/misfit()/rss() and the
    // rest of the forward-solver observers describe the fitted model
    void write_state_(const vector_t& z) {
        Y_ = forward_(z);
        U_ = matrix_t::Zero(m_ - 1, d_);   // NLS stays on the model manifold: no control
        bound_mult_.setZero(m_, d_);
        compute_misfit_();
        flatten_();
        objective_value_ = S_(z);
        n_iter_ = outer_n_iter_;
        converged_ = outer_converged_;
        return;
    }

    // objective functor over the decision vector, adapting S and its adjoint gradient to the core BFGS
    // interface (the same shape the tracking solver's outer criterion presents)
    struct nls_objective {
        ts_ls_ode_nls* solver;
        double operator()(const vector_t& z) const { return solver->S_(z); }
        auto gradient() const {
            return [s = solver](const vector_t& z) { return s->grad_S_(z); };
        }
    };

    std::function<matrix_t(const vector_t&)> ic_jacobian_;
    std::function<vector_t(const vector_t&)> ic_from_theta_;

    vector_t theta_;                 // current / estimated parameters
    bool estimate_ic_ = false;       // is the initial state a decision variable?
    int outer_max_iter_ = 100;
    double outer_tol_ = 1e-6;
    line_search line_search_ = line_search::backtracking;

    // one-slot forward cache
    vector_t z_cache_;
    matrix_t Yz_;
    int n_finite_ = 0;              // length of the finite prefix of the cached trajectory (== m_ if complete)
    bool fwd_valid_z_ = false;
    int n_divergent_ = 0;

    // best evaluated point, the fallback solve() returns when the optimizer's last iterate is not usable
    vector_t best_z_;
    double best_value_ = std::numeric_limits<double>::infinity();
    bool safeguarded_ = false;

    // results
    double outer_value_ = 0;
    int outer_n_iter_ = 0;
    bool outer_converged_ = false;
};

}   // namespace internals
}   // namespace fdapde

#endif   // __TS_LS_ODE_NLS_SOLVER_H__
