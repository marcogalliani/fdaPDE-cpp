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

#ifndef __BS_LS_ODE_NLS_SOLVER_H__
#define __BS_LS_ODE_NLS_SOLVER_H__

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
// WHY IT EXISTS. It is the reference estimator the tracking solver (bs_ls_ode_tracking) is to be measured
// against. The tracking criterion H_lambda(theta) = min_u J(u, theta) admits a control u that lets the
// trajectory leave the model manifold, and the claim motivating it (Ramsay & Hooker's generalized
// profiling) is precisely that H_lambda is a SMOOTHED version of S, approaching it as lambda -> infinity.
// Testing that claim requires S itself, computed on the SAME model, the SAME time grid and the SAME
// Butcher tableau -- otherwise a comparison of estimators is partly a comparison of discretizations.
// Hence this solver consumes the same bs_ls_ode_param descriptor and the same erased engine as the
// tracking solver, and derives from the same bs_ls_ode base so data ingestion, the observation mask and
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
class bs_ls_ode_nls : public bs_ls_ode {
   public:
    using Base = bs_ls_ode;
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

    bs_ls_ode_nls() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    bs_ls_ode_nls(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty);
        analyze_data(formula, gf);
    }

    // discretize a parametric descriptor (bs_ls_ode_param): the base stores the erased engine at theta0
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
    // Run the outer BFGS on z, theta_k = lower_k + exp(z_k), instead of on theta directly (see
    // theta_log_map). `lower` has one entry per PARAMETER; a NaN entry leaves that component unmapped,
    // and an estimated initial state is never mapped. The estimate and every observer stay in theta.
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
    // which optimizer minimizes S: BFGS with a line search (the default) or the trust-region
    // quasi-Newton, which rejects a step into the divergent region instead of shortening it
    // levenberg_marquardt is a TRUE Gauss-Newton method here: S is a sum of squares and the residual
    // Jacobian d y/d z is available exactly, by forward propagation of the same step Jacobians the
    // adjoint sweep contracts backwards (see nls_ls_problem).
    enum class outer_method { bfgs, trust_region, levenberg_marquardt };
    void set_outer_method(outer_method m) { outer_method_ = m; }
    outer_method outer_method_policy() const { return outer_method_; }
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
        // the optimizer's variables are w (the decision vector itself under the identity map)
        const vector_t w0 = w_of_z_(decision_vector());
        vector_t z_opt;
        if (outer_method_ == outer_method::levenberg_marquardt) {
            LevenbergMarquardt<Dynamic> lm(outer_max_iter_, outer_tol_);
            z_opt = z_of_w_(lm.optimize(nls_ls_problem {this}, w0));
            outer_n_iter_ = lm.n_iter();
            outer_converged_ = lm.converged();
        } else if (outer_method_ == outer_method::trust_region) {
            TrustRegion<Dynamic> tr(outer_max_iter_, outer_tol_, outer_step_cap_ > 0 ? outer_step_cap_ : 1.0);
            z_opt = z_of_w_(tr.optimize(problem, w0));
            outer_n_iter_ = tr.n_iter();
            outer_converged_ = tr.converged();
        } else {
            z_opt = z_of_w_(
              (line_search_ == line_search::wolfe) ?
                optimizer.optimize(problem, w0, WolfeLineSearch(), step_norm_cap {outer_step_cap_}) :
                optimizer.optimize(problem, w0, BacktrackingLineSearch(), step_norm_cap {outer_step_cap_}));
            outer_n_iter_ = optimizer.n_iter();
            outer_converged_ = optimizer.converged();
        }
        // the optimizer's own last iterate, kept before the safeguard may replace it
        theta_last_ = theta_of_(z_opt);
        outer_value_last_ = S_(z_opt);
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
        write_state_(z_opt);   // leave the forward state consistent with the estimate
        return theta_;
    }
    const vector_t& solve() { return solve(theta_); }

    // --- observers -----------------------------------------------------
    // named as in bs_ls_ode_tracking, so a driver reports both estimators through the same calls. There
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
    // the optimizer's own last iterate and S there, before the safeguard: reporting a failure rate on both
    // this and the returned estimate separates a solved problem from a rescued one
    const vector_t& theta_last_iterate() const { return theta_last_; }
    double outer_objective_last_iterate() const { return outer_value_last_; }

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
    // DIAGNOSTIC: the gradient as the least-squares view computes it, 2 J^T r. It must equal
    // gradient_at() to round-off -- the forward sensitivity propagation and the backward adjoint sweep
    // contract the same step Jacobians in opposite orders -- which is the check on the Jacobian LM uses.
    // Takes the OPTIMIZER's variables w (equal to z when no parameter map is set).
    vector_t ls_gradient_at(const vector_t& w) {
        nls_ls_problem p {this};
        return 2.0 * (p.jacobian(w).transpose() * p.residuals(w));
    }
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
        Yz_ = engine_.solve(time_, y0_of_(z, theta));
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
    //     S = sum_{t < k} m_t (y_t - y^obs_t)^2 + divergent_prefix_cost_ * (m - k)/m,   k = finite prefix length.
    //
    // Two properties matter. (i) Any divergent point costs at least divergent_prefix_cost_/m, which dominates any
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
        if (!std::isfinite(s)) { return divergent_prefix_cost_; }
        if (n_finite_ < m_) { return s + divergent_prefix_cost_ * double(m_ - n_finite_) / m_; }
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
        const matrix_t no_control = matrix_t::Zero(d_, s_);   // one zero column per stage: NLS forces nothing
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
        U_ = matrix_t::Zero(m_ - 1, sd_());   // NLS stays on the model manifold: no control
        bound_mult_.setZero(m_, d_);
        capture_dynamics_();   // this solver bypasses both base fit paths, so it captures its own
        compute_misfit_();
        flatten_();
        objective_value_ = S_(z);
        n_iter_ = outer_n_iter_;
        converged_ = outer_converged_;
        return;
    }

    // objective functor over the decision vector, adapting S and its adjoint gradient to the core BFGS
    // interface (the same shape the tracking solver's outer criterion presents)
    // decision vector z = [theta; y_0] <-> optimizer variables w: the map acts on the theta head only, an
    // estimated initial state is a plain unconstrained variable
    vector_t z_of_w_(const vector_t& w) const {
        if (!param_map_.active()) { return w; }
        const int n_theta = engine_.n_params();
        vector_t z = w;
        z.head(n_theta) = param_map_.theta_of(vector_t(w.head(n_theta)));
        return z;
    }
    vector_t w_of_z_(const vector_t& z) const {
        if (!param_map_.active()) { return z; }
        const int n_theta = engine_.n_params();
        vector_t w = z;
        w.head(n_theta) = param_map_.z_of(vector_t(z.head(n_theta)));
        return w;
    }
    vector_t chain_(const vector_t& z, const vector_t& grad) const {
        if (!param_map_.active()) { return grad; }
        const int n_theta = engine_.n_params();
        vector_t g = grad;
        g.head(n_theta) = param_map_.chain(vector_t(z.head(n_theta)), vector_t(grad.head(n_theta)));
        return g;
    }

    /* The SAME criterion S, presented as residuals and a Jacobian so Levenberg-Marquardt can build the
    Gauss-Newton curvature J^T J exactly instead of accumulating it from secant pairs.

    LAYOUT. One entry per observed (node, component) pair, in observed_entries_() order, plus ONE trailing
    entry sqrt(divergent_prefix_cost_ * (m - k)/m) carrying the partial-prefix penalty S_ applies when the
    integration blew up after k < m nodes. With that entry, ||r||^2 == S_(z) exactly, so LM minimises the
    very same function BFGS and the trust region do and the three are directly comparable. Entries past
    the finite prefix are zero, as are their Jacobian rows -- the same convention grad_S_ follows, which
    sums over the prefix only, so the penalty contributes to the value and not to the derivative.

    CHECK. 2 J^T r is grad_S_ to round-off by construction; the driver's --gradcheck exercises it. */
    struct nls_ls_problem {
        bs_ls_ode_nls* s;
        Eigen::Matrix<double, Dynamic, 1> residuals(const vector_t& w) const {
            const vector_t z = s->z_of_w_(w);
            const matrix_t& Y = s->forward_(z);
            const auto& idx = s->obs_idx_();
            Eigen::Matrix<double, Dynamic, 1> r =
              Eigen::Matrix<double, Dynamic, 1>::Zero(static_cast<int>(idx.size()) + 1);
            const int k = s->n_finite_;
            for (std::size_t i = 0; i < idx.size(); ++i) {
                const auto [t, v] = idx[i];
                if (t < k) { r[static_cast<int>(i)] = Y(t, v) - s->y_obs_(t, v); }
            }
            if (k < s->m_) {
                r[static_cast<int>(idx.size())] =
                  std::sqrt(divergent_prefix_cost_ * static_cast<double>(s->m_ - k) / s->m_);
            }
            return r;
        }
        Eigen::Matrix<double, Dynamic, Dynamic> jacobian(const vector_t& w) const {
            const vector_t z = s->z_of_w_(w);
            const matrix_t& Y = s->forward_(z);
            const auto& idx = s->obs_idx_();
            const int q = static_cast<int>(z.size()), n_theta = s->engine_.n_params();
            Eigen::Matrix<double, Dynamic, Dynamic> J =
              Eigen::Matrix<double, Dynamic, Dynamic>::Zero(static_cast<int>(idx.size()) + 1, q);
            // seed: d y_0 / d z. A jointly estimated initial state is the identity in its own block; a
            // hard IC that depends on theta contributes its Jacobian; a fixed hard IC contributes nothing.
            matrix_t S0 = matrix_t::Zero(s->d_, q);
            if (s->estimate_ic_) { S0.rightCols(s->d_) = matrix_t::Identity(s->d_, s->d_); }
            else if (s->has_ic_ && s->ic_jacobian_) { S0.leftCols(n_theta) = s->ic_jacobian_(s->theta_of_(z)); }
            const auto S = s->param_sensitivity_(Y, nullptr, S0, s->n_finite_);
            for (std::size_t i = 0; i < idx.size(); ++i) {
                const auto [t, v] = idx[i];
                if (t < static_cast<int>(S.size())) { J.row(static_cast<int>(i)) = S[t].row(v); }
            }
            // chain through the parameter map: theta_j = lower_j + exp(w_j) scales column j by dtheta/dw
            if (s->param_map_.active()) {
                const vector_t th = s->theta_of_(z);
                const vector_t d = s->param_map_.chain(th, vector_t::Ones(n_theta));
                for (int j = 0; j < n_theta; ++j) { J.col(j) *= d[j]; }
            }
            return J;
        }
    };
    // the observed (node, component) pairs, cached: the residual layout must be stable across iterations
    const std::vector<std::pair<int, int>>& obs_idx_() {
        if (obs_idx_cache_.empty() && n_obs_ > 0) { obs_idx_cache_ = observed_entries_(); }
        return obs_idx_cache_;
    }
    std::vector<std::pair<int, int>> obs_idx_cache_;

    struct nls_objective {
        bs_ls_ode_nls* solver;
        double operator()(const vector_t& w) const { return solver->S_(solver->z_of_w_(w)); }
        auto gradient() const {
            return [s = solver](const vector_t& w) {
                const vector_t z = s->z_of_w_(w);
                return s->chain_(z, s->grad_S_(z));   // grad_S_ already zeroes a non-finite gradient
            };
        }
    };

    std::function<matrix_t(const vector_t&)> ic_jacobian_;
    std::function<vector_t(const vector_t&)> ic_from_theta_;

    vector_t theta_;                 // current / estimated parameters
    bool estimate_ic_ = false;       // is the initial state a decision variable?
    int outer_max_iter_ = 100;
    double outer_tol_ = 1e-6;
    vector_t theta_last_;         // the optimizer's last iterate, before the safeguard
    double outer_value_last_ = std::numeric_limits<double>::infinity();
    outer_method outer_method_ = outer_method::bfgs;
    theta_log_map param_map_;     // identity unless set_parameter_map() was called
    double outer_step_cap_ = 0;   // <= 0: the step length is left to the line search
    // Magnitude of the horizon-loss term in the partial-prefix cost S_ scores a blown-up shooting solve
    // with. Unlike the base solver's flat sentinel (bs_ls_ode::divergent_cost_, which is infinity) this
    // one enters an ARITHMETIC expression -- s + divergent_prefix_cost_ * (m - k)/m -- whose whole point
    // is to stay finite so that reaching further before blowing up scores strictly better. Infinity here
    // would collapse every divergent point onto the same value and destroy that ordering.
    static constexpr double divergent_prefix_cost_ = 1e20;
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

#endif   // __BS_LS_ODE_NLS_SOLVER_H__
