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

#ifndef __BS_LS_ODE_H1_SOLVER_H__
#define __BS_LS_ODE_H1_SOLVER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

/* H1-penalized tracking solver.
===============================================================================================
Same estimator as bs_ls_ode_tracking -- minimize over theta the value of an inner optimal-control
problem -- with the norm on the model misfit changed from L2 to H1:

    H^{H1}_lambda(theta) = min_u  sum_i |y(t_i; u, theta) - y_i|^2
                                  + lambda * int ( ||u||^2 + l^2 ||u'||^2 ) dt,

against the tracking solver's  lambda * int ||u||^2 dt.  Here u is the additive control forcing the
prior dynamics y' = f(t, y, theta) + u, i.e. the model misfit itself, and l is a LENGTH (set by
set_penalty_length, default 1).

WHY. Linearizing the residuals (docs/penalty-smoothing.md 2), the Gauss-Newton Hessian of the L2
criterion is

    grad^2 H_lambda = 2 G^T (K + lambda^{-1} I)^{-1} G,      K = C Q^{-1} C^T,  C = d/dt - A,  G = df/dtheta,

a Tikhonov RIDGE of size 1/lambda on the propagator. Replacing the L2 norm on u by the H1 norm
<u, M u> with M = I - l^2 d^2/dt^2 replaces that ridge by an OPERATOR one:

    grad^2 H^{H1}_lambda = 2 Gt^T (Kt + lambda^{-1} I)^{-1} Gt,    Gt = M^{1/2} G,  Kt = M^{1/2} K M^{1/2}.

The practical difference is the memory length of the smoothing. The control is the solution of a
Riccati/LQR problem whose Green's function decays over

    tau_L2  ~ (lambda / w)^{1/2}        (L2 penalty, per docs/penalty-smoothing.md 3)
    tau_H1  ~ (lambda l^2 / w)^{1/4}    (H1 penalty),

so at the same lambda the H1 penalty has a SHORTER memory but buys it with a control that cannot
oscillate: the l^2 ||u'||^2 term charges roughness directly, which is what an L2 penalty cannot do at
any lambda (it charges only size). Whether that trades favourably is an empirical question -- which is
what simulations/nls-vs-tracking/h1-penalty exists to answer -- not something the algebra settles.

IMPLEMENTATION. Self-contained: it derives from bs_ls_ode and touches no other solver.

  * INNER. fit() below replaces the inherited one. The base's reduced adjoint solve already carries the
    L2 part of the penalty exactly (it is diagonal in the scaled decision variable), so only the
    derivative term is added, to both the objective and the gradient. Discretely, with the control's
    stage points tau_0 < ... < tau_{N-1} in time order (N = (m-1)*s of them, tau at time_(t) + c_i dt_t),

        int ||u'||^2 dt  ~  sum_j ||u_{j+1} - u_j||^2 / (tau_{j+1} - tau_j),

    a tridiagonal quadratic form. This is the right discretization for THIS control space: the control
    is a BROKEN polynomial, discontinuous at the interval boundaries, so an exact within-interval
    derivative would (i) miss the jumps, which is where the roughness lives, and (ii) vanish identically
    for the implicit-midpoint scheme (s = 1, u constant per interval). The difference quotient across
    consecutive stage points charges both.

  * PRECONDITIONER. The base scales each decision block by s_blk ~ sqrt(diagonal of the inner Hessian);
    the H1 term adds 2 lambda l^2 (1/Delta_{j-1} + 1/Delta_j) to that diagonal, which at l = 1 and this
    project's grids is one to two orders of magnitude above the L2 contribution. build_h1_() folds it in
    (and rescales ctrl_w_ so the L2 part of the objective is unchanged), otherwise the inner BFGS spends
    its budget discovering the scaling it was handed for free.

  * OUTER. Unchanged from the tracking solver, and deliberately so: the envelope theorem makes the outer
    gradient the PARTIAL derivative of the inner objective at the inner optimum, and the penalty --
    whatever its norm -- carries no theta. So grad H is the same single backward adjoint sweep at frozen
    control. The outer loop below is that of bs_ls_ode_tracking (parameter map, trust region or BFGS,
    best-point safeguard), duplicated rather than inherited because the inner solve is not virtual there.

  * POLICY. Only the reduced adjoint inner solve is implemented. The SQP policy carries the L2 penalty in
    its KKT objective, which the H1 form would have to rewrite; a study comparing the two penalties must
    therefore put the L2 arm on the adjoint policy too, which is what --inner adjoint does.
*/
class bs_ls_ode_h1 : public bs_ls_ode {
   public:
    using Base = bs_ls_ode;

    bs_ls_ode_h1() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    bs_ls_ode_h1(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty);
        analyze_data(formula, gf);
    }

    // discretize a parametric penalty (a bs_ls_ode_param packet), exactly as the tracking solver does
    template <typename Penalty> void discretize(Penalty&& penalty) {
        Base::discretize(penalty);
        theta_ = penalty.theta0();
        fdapde_assert(static_cast<bool>(engine_) && theta_.size() == engine_.n_params());
        invalidate_cache_();
    }

    // --- configuration -------------------------------------------------
    // the length l in  lambda * int ( ||u||^2 + l^2 ||u'||^2 ).  l = 0 recovers the L2 penalty exactly
    // (and then this solver is bs_ls_ode_tracking on the adjoint policy). Default 1.
    void set_penalty_length(double ell) {
        fdapde_assert(ell >= 0);
        ell_ = ell;
        invalidate_cache_();
    }
    double penalty_length() const { return ell_; }
    void set_outer_options(int max_iter, double tol) {
        fdapde_assert(max_iter > 0 && tol > 0);
        outer_max_iter_ = max_iter;
        outer_tol_ = tol;
    }
    // run the outer solve on z, theta_k = lower_k + exp(z_k) (see theta_log_map); NaN leaves a component
    // unmapped. The estimate and every observer stay in theta.
    void set_parameter_map(const vector_t& lower) {
        fdapde_assert(static_cast<bool>(engine_) && lower.size() == engine_.n_params());
        param_map_.lower = lower;
    }
    void clear_parameter_map() { param_map_.lower.resize(0); }
    const theta_log_map& parameter_map() const { return param_map_; }
    void set_outer_step_cap(double cap) { outer_step_cap_ = cap; }
    // levenberg_marquardt here is Gauss-Newton on the FROZEN-CONTROL residual Jacobian: the gradient
    // it uses is exact (the envelope theorem), the curvature is an upper bound. See h1_ls_problem.
    enum class outer_method { bfgs, trust_region, levenberg_marquardt };
    void set_outer_method(outer_method m) { outer_method_ = m; }
    outer_method outer_method_policy() const { return outer_method_; }
    // LM only: the exact reduced (variable-projection) Jacobian, the default, or the frozen-control one
    void use_exact_jacobian(bool on) { exact_jacobian_ = on; }
    bool exact_jacobian() const { return exact_jacobian_; }
    void set_ic_jacobian(std::function<matrix_t(const vector_t&)> ic_jacobian) {
        ic_jacobian_ = std::move(ic_jacobian);
    }
    void set_ic_parameterization(
      std::function<vector_t(const vector_t&)> ic, std::function<matrix_t(const vector_t&)> ic_jac) {
        fdapde_assert(has_ic_ && static_cast<bool>(ic) && static_cast<bool>(ic_jac));
        ic_from_theta_ = std::move(ic);
        ic_jacobian_ = std::move(ic_jac);
        invalidate_cache_();
    }

    // --- inner solve ----------------------------------------------------
    /* min_u  SSE(observed) + lambda * int ( ||u||^2 + l^2 ||u'||^2 ),  the state integrated out.
    This HIDES bs_ls_ode::fit (both overloads) on purpose: on this class a plain fit() must be the H1
    problem, or the outer criterion and the inner one -- which the envelope theorem requires to be the
    same functional -- silently disagree. */
    const vector_t& fit(double lambda) {
        fdapde_assert(lambda > 0 && d_ > 0 && m_ >= 2 && static_cast<bool>(engine_));
        if (has_ic_) { fdapde_assert(y0_.size() == d_); }
        lambda_ = lambda;
        build_ctrl_scale_();   // the base's L2 preconditioner ...
        build_h1_();           // ... plus the H1 stencil and its share of the Hessian diagonal
        fwd_valid_ = false;
        has_best_ = false;
        coeff_valid_ = false;
        const int sd = sd_(), nu = n_ctrl_();
        const bool free_y1 = !has_ic_;
        vector_t z = vector_t::Zero(nu + (free_y1 ? d_ : 0));
        if (free_y1) { z.segment(nu, d_) = initial_guess_().row(0).transpose(); }
        // the u = 0 fit of the prior dynamics: seeds the best-iterate tracker and is the floor the guard
        // below tests against (see the same construction in bs_ls_ode::fit_adjoint_)
        const double J_start = objective_h1_(z);
        if (std::isfinite(J_start)) {
            best_z_ = z;
            best_J_ = J_start;
            has_best_ = true;
        }
        h1_objective problem {this};
        BFGS<Dynamic> optimizer(max_iter_, tol_, 1.0);
        vector_t z_opt = optimizer.optimize(problem, z, WolfeLineSearch<15>(), best_iterate {this});
        if (has_best_ && !(objective_h1_(z_opt) <= J_start)) { z_opt = best_z_; }
        Y_ = forward_recover_(z_opt);
        if (has_ic_) { Y_.row(0) = y0_.transpose(); }
        U_.resize(m_ - 1, sd);
        for (int t = 0; t < m_ - 1; ++t) { set_stage_block_(U_, t, expand_(z_opt, t)); }
        bound_mult_.setZero(m_, d_);   // the adjoint policy ignores state bounds
        objective_value_ = objective_h1_(z_opt);
        n_iter_ = optimizer.n_iter();
        converged_ = Y_.allFinite() && gradient_h1_(z_opt).norm() <= tol_;
        capture_dynamics_();
        compute_misfit_();
        flatten_();
        return f();
    }

    // --- outer solve ----------------------------------------------------
    const vector_t& solve(double lambda, const vector_t& theta0) {
        fdapde_assert(lambda > 0 && static_cast<bool>(engine_));
        fdapde_assert(theta0.size() == engine_.n_params());
        lambda_outer_ = lambda;
        inner_failures_ = 0;
        best_value_ = std::numeric_limits<double>::infinity();
        best_theta_.resize(0);
        invalidate_cache_();
        h1_outer_objective problem {this};
        const vector_t z0 = param_map_.z_of(theta0);
        vector_t z_opt;
        if (outer_method_ == outer_method::levenberg_marquardt) {
            LevenbergMarquardt<Dynamic> lm(outer_max_iter_, outer_tol_);
            z_opt = lm.optimize(h1_ls_problem {this}, z0);
            outer_n_iter_ = lm.n_iter();
            outer_converged_ = lm.converged();
        } else if (outer_method_ == outer_method::trust_region) {
            TrustRegion<Dynamic> optimizer(outer_max_iter_, outer_tol_, outer_step_cap_ > 0 ? outer_step_cap_ : 1.0);
            z_opt = optimizer.optimize(problem, z0);
            outer_n_iter_ = optimizer.n_iter();
            outer_converged_ = optimizer.converged();
        } else {
            BFGS<Dynamic> optimizer(outer_max_iter_, outer_tol_, 1.0);
            z_opt = optimizer.optimize(problem, z0, WolfeLineSearch(), step_norm_cap {outer_step_cap_});
            outer_n_iter_ = optimizer.n_iter();
            outer_converged_ = optimizer.converged();
        }
        vector_t theta_opt = param_map_.theta_of(z_opt);
        theta_last_ = theta_opt;   // the optimizer's own last iterate, before the safeguard below
        outer_value_last_ = theta_opt.allFinite() ? H_(theta_opt) : std::numeric_limits<double>::infinity();
        // SAFEGUARD: return the best theta actually EVALUATED rather than the last iterate (see
        // bs_ls_ode_tracking::solve for why the last iterate can be unusable)
        if (!theta_opt.allFinite() || outer_value_last_ > best_value_) {
            fdapde_assert(best_theta_.size() > 0);
            theta_opt = best_theta_;
            safeguarded_ = true;
        } else {
            safeguarded_ = false;
        }
        theta_ = theta_opt;
        invalidate_cache_();
        outer_value_ = H_(theta_);   // leaves the forward state consistent with the estimate
        return theta_;
    }
    const vector_t& solve(double lambda) { return solve(lambda, theta_); }

    // --- observers ------------------------------------------------------
    const vector_t& theta() const { return theta_; }
    int n_params() const { return engine_.n_params(); }
    double outer_objective() const { return outer_value_; }
    int outer_n_iter() const { return outer_n_iter_; }
    bool outer_converged() const { return outer_converged_; }
    int n_inner_failures() const { return inner_failures_; }
    bool safeguarded() const { return safeguarded_; }
    const vector_t& theta_last_iterate() const { return theta_last_; }
    double outer_objective_last_iterate() const { return outer_value_last_; }
    // the two halves of the penalty AT THE CURRENT FIT, in physical control units and without lambda:
    //   int ||u||^2 dt   and   int ||u'||^2 dt  (the same discrete forms the objective uses).
    // Reported side by side they are the direct evidence of what the H1 term bought -- a control of
    // comparable size but lower roughness -- and control_roughness() is defined for any solver of this
    // family, so a study can compute it for the L2 arm from its control() and compare.
    double control_energy() const { return control_energy_(U_); }
    double control_roughness() const { return control_roughness_(U_); }

    /* DIAGNOSTIC: the INNER objective and its gradient at an explicit control.
    z is in the scaled decision variables the inner solve works in -- inner_decision_size() long, with
    the free-IC tail when there is no hard initial condition -- and prepare_inner(lambda) must be called
    first to build the preconditioner and the H1 stencil at that lambda. Exposed so the derivative term
    can be finite-difference checked: it is the one piece of new math in this class, the rest being the
    base's adjoint solve and the tracking solver's envelope gradient. */
    int inner_decision_size() const { return n_ctrl_() + (has_ic_ ? 0 : d_); }
    void prepare_inner(double lambda) {
        fdapde_assert(lambda > 0 && d_ > 0 && m_ >= 2 && static_cast<bool>(engine_));
        lambda_ = lambda;
        build_ctrl_scale_();
        build_h1_();
        fwd_valid_ = false;
    }
    double inner_objective_at(const vector_t& z) { return objective_h1_(z); }
    vector_t inner_gradient_at(const vector_t& z) { return gradient_h1_(z); }

    double outer_objective_at(double lambda, const vector_t& theta) {
        set_lambda_(lambda);
        return H_(theta);
    }
    vector_t outer_gradient_at(double lambda, const vector_t& theta) {
        set_lambda_(lambda);
        return grad_H_(theta);
    }
    // DIAGNOSTIC: 2 J^T r from the least-squares view; must equal the envelope gradient above. See the
    // note on h1_ls_problem for why the gradient is exact while the curvature is only an upper bound.
    vector_t ls_gradient_at(double lambda, const vector_t& z) {
        set_lambda_(lambda);
        h1_ls_problem p {this};
        return 2.0 * (p.jacobian(z).transpose() * p.residuals(z));
    }

   private:
    // ---- the H1 stencil ------------------------------------------------
    // stage points in time order and the gaps between them; plus the H1 share of the inner Hessian
    // diagonal, folded into the base's preconditioner. Call after build_ctrl_scale_(), once per fit.
    void build_h1_() {
        const int nb = n_blocks_();
        const vector_t tau = control_stage_times();   // the control's abscissae, in time order
        gap_.assign(nb > 1 ? nb - 1 : 0, 1.0);
        for (int j = 0; j + 1 < nb; ++j) { gap_[j] = std::max(tau[j + 1] - tau[j], gap_floor_); }
        if (ell_ <= 0) { return; }
        const vector_t b = quad_weights_();
        for (int t = 0; t < m_ - 1; ++t) {
            for (int i = 0; i < s_; ++i) {
                const int j = t * s_ + i;
                // the L2 block of the inner Hessian, which build_ctrl_scale_ already used
                const double pen_l2 = 2.0 * lambda_ * dt_(t) * b[i];
                // the H1 block: d^2/du_j^2 of lambda l^2 sum ||u_{j+1} - u_j||^2 / Delta
                const double pen_h1 = 2.0 * lambda_ * ell_ * ell_ *
                                      ((j > 0 ? 1.0 / gap_[j - 1] : 0.0) + (j + 1 < nb ? 1.0 / gap_[j] : 0.0));
                const double s_new = std::sqrt(ctrl_s_[j] * ctrl_s_[j] + pen_h1);
                ctrl_s_[j] = s_new;
                // ctrl_w_ is defined so that the base's 0.5 w ||z_blk||^2 equals lambda dt b ||u_blk||^2;
                // rescaling s_blk therefore requires rescaling w_blk, or the L2 penalty changes with the
                // preconditioner
                ctrl_w_[j] = pen_l2 / (s_new * s_new);
            }
        }
        return;
    }
    // l^2 int ||u'||^2 dt in the SCALED decision variable (u_j = z_j / s_j), times lambda
    double h1_term_(const vector_t& z) const {
        if (ell_ <= 0) { return 0.0; }
        const int nb = n_blocks_();
        double P = 0;
        for (int j = 0; j + 1 < nb; ++j) {
            const vector_t du =
              z.segment((j + 1) * d_, d_) / ctrl_s_[j + 1] - z.segment(j * d_, d_) / ctrl_s_[j];
            P += du.squaredNorm() / gap_[j];
        }
        return lambda_ * ell_ * ell_ * P;
    }
    double objective_h1_(const vector_t& z) {
        const double J = objective_(z);   // data term + the L2 half of the penalty
        return std::isfinite(J) ? J + h1_term_(z) : J;
    }
    vector_t gradient_h1_(const vector_t& z) {
        vector_t g = gradient_(z);
        // a divergent forward pass: the base returns a finite ZERO gradient and an infinite objective, so
        // the line search backtracks on the objective alone. Leave that contract intact.
        if (ell_ <= 0 || !forward_recover_(z).allFinite()) { return g; }
        const int nb = n_blocks_();
        const double c = 2.0 * lambda_ * ell_ * ell_;
        for (int j = 0; j + 1 < nb; ++j) {
            const vector_t du =
              (z.segment((j + 1) * d_, d_) / ctrl_s_[j + 1] - z.segment(j * d_, d_) / ctrl_s_[j]) * (c / gap_[j]);
            g.segment(j * d_, d_) -= du / ctrl_s_[j];
            g.segment((j + 1) * d_, d_) += du / ctrl_s_[j + 1];
        }
        return g;
    }
    // the same two quadratures on a physical control schedule ((m-1) x (s*d), stage-major)
    double control_energy_(const matrix_t& U) const {
        if (U.rows() != m_ - 1 || U.cols() != sd_()) { return std::numeric_limits<double>::quiet_NaN(); }
        const vector_t b = quad_weights_();
        double E = 0;
        for (int t = 0; t < m_ - 1; ++t) {
            for (int i = 0; i < s_; ++i) { E += dt_(t) * b[i] * U.block(t, i * d_, 1, d_).squaredNorm(); }
        }
        return E;
    }
    double control_roughness_(const matrix_t& U) const {
        const int nb = n_blocks_();
        if (U.rows() != m_ - 1 || U.cols() != sd_() || static_cast<int>(gap_.size()) + 1 != nb) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        auto blk = [&](int j) { return U.block(j / s_, (j % s_) * d_, 1, d_); };
        double P = 0;
        for (int j = 0; j + 1 < nb; ++j) { P += (blk(j + 1) - blk(j)).squaredNorm() / gap_[j]; }
        return P;
    }
    // inner objective functor over the control z, for the core BFGS interface
    struct h1_objective {
        bs_ls_ode_h1* solver;
        double operator()(const vector_t& z) const { return solver->objective_h1_(z); }
        auto gradient() const {
            return [s = solver](const vector_t& z) { return s->gradient_h1_(z); };
        }
    };

    // ---- the outer problem (as bs_ls_ode_tracking) ----------------------
    void inner_solve_(const vector_t& theta) {
        if (cache_valid_ && theta_cache_.size() == theta.size() && theta_cache_ == theta) { return; }
        engine_.set_theta(theta);
        if (ic_from_theta_) { y0_ = ic_from_theta_(theta); }
        fit(lambda_outer_);
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
            invalidate_cache_();
        }
        return;
    }
    double H_(const vector_t& theta) {
        inner_solve_(theta);
        double v = objective();
        if (!std::isfinite(v) || !Y_.allFinite()) { v = divergent_outer_cost_; }
        if (v < best_value_ && theta.allFinite()) {
            best_value_ = v;
            best_theta_ = theta;
        }
        return v;
    }
    /* Envelope gradient: one backward adjoint sweep at FROZEN control.
    Identical to the tracking solver's, and that is the point -- the inner optimality condition dJ/du = 0
    collapses the total derivative onto the partial one, and the penalty (L2 or H1, it makes no difference)
    carries no theta, so only the data term survives. Changing the penalty's norm changes WHERE the inner
    solve stops, not the formula for the outer gradient there. */
    vector_t grad_H_(const vector_t& theta) {
        inner_solve_(theta);
        vector_t grad = vector_t::Zero(engine_.n_params());
        if (!Y_.allFinite() || !U_.allFinite()) { return grad; }
        auto source = [&](int t) {
            vector_t s = vector_t::Zero(d_);
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { s(v) = 2.0 * (Y_(t, v) - y_obs_(t, v)); }
            }
            return s;
        };
        vector_t p = source(m_ - 1);
        for (int t = m_ - 2; t >= 0; --t) {
            vector_t yc = Y_.row(t).transpose();
            const matrix_t ut = stage_block_(U_, t);
            auto s = engine_.step_with_flow_param_jacobians(time_(t), yc, dt_(t), ut);
            grad.noalias() += s.param.transpose() * p;
            p = source(t) + s.flow.transpose() * p;
        }
        if (has_ic_ && ic_jacobian_) { grad.noalias() += ic_jacobian_(theta).transpose() * p; }
        return grad;
    }
    /* The H1-penalized criterion as residuals and a Jacobian for Levenberg-Marquardt.

    Identical in structure to bs_ls_ode_tracking::tracking_ls_problem -- read the derivation there -- with
    the residual vector carrying the H1 penalty's two blocks,

        R = [ y - y_obs ; sqrt(lambda dt_t b_i) u_j ; sqrt(lambda l^2 / Delta_j) (u_{j+1} - u_j) ] ,

    so ||R||^2 == H^{H1}_lambda exactly. The only thing that changes in the variable projection is
    M = P^T P, which is DIAGONAL for the L2 penalty and TRIDIAGONAL here: the derivative block
    contributes lambda l^2 times the weighted graph Laplacian of the chain of stage points. A sparse
    Cholesky handles either, so the Woodbury reduction to an n_obs x n_obs solve is unaffected. */
    struct h1_ls_problem {
        bs_ls_ode_h1* s;
        Eigen::Matrix<double, Dynamic, 1> residuals(const vector_t& z) const {
            const vector_t theta = s->param_map_.theta_of(z);
            s->inner_solve_(theta);
            const auto& idx = s->obs_idx_();
            const int n_obs = static_cast<int>(idx.size()), nu = s->n_ctrl_(), nb = s->n_blocks_();
            const int n_der = (nb > 1 && s->ell_ > 0) ? (nb - 1) * s->d_ : 0;
            Eigen::Matrix<double, Dynamic, 1> r(n_obs + nu + n_der);
            if (!s->Y_.allFinite() || !s->U_.allFinite() || !std::isfinite(s->objective())) {
                return r.setConstant(std::numeric_limits<double>::quiet_NaN());
            }
            for (int i = 0; i < n_obs; ++i) {
                const auto [t, v] = idx[i];
                r[i] = s->Y_(t, v) - s->y_obs_(t, v);
            }
            const vector_t u = s->flat_control_();
            r.segment(n_obs, nu) = s->penalty_weight_().cwiseSqrt().cwiseProduct(u);
            for (int j = 0; j + 1 < nb && n_der > 0; ++j) {
                const double c = std::sqrt(s->lambda_outer_ * s->ell_ * s->ell_ / s->gap_[j]);
                r.segment(n_obs + nu + j * s->d_, s->d_) =
                  c * (u.segment((j + 1) * s->d_, s->d_) - u.segment(j * s->d_, s->d_));
            }
            return r;
        }
        Eigen::Matrix<double, Dynamic, Dynamic> jacobian(const vector_t& z) const {
            const vector_t theta = s->param_map_.theta_of(z);
            s->inner_solve_(theta);
            const auto& idx = s->obs_idx_();
            const int n_obs = static_cast<int>(idx.size()), nu = s->n_ctrl_(), nb = s->n_blocks_();
            const int p = s->engine_.n_params(), d = s->d_;
            const int n_der = (nb > 1 && s->ell_ > 0) ? (nb - 1) * d : 0;
            Eigen::Matrix<double, Dynamic, Dynamic> J =
              Eigen::Matrix<double, Dynamic, Dynamic>::Zero(n_obs + nu + n_der, p);
            if (!s->Y_.allFinite() || !s->U_.allFinite()) { return J; }
            matrix_t S0 = matrix_t::Zero(d, p);
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
                // M = P^T P = diag(w) + lambda l^2 * (chain Laplacian with edge weights 1/Delta_j)
                std::vector<Eigen::Triplet<double>> trip;
                trip.reserve(nu + 4 * n_der);
                for (int j = 0; j < nu; ++j) { trip.emplace_back(j, j, w[j]); }
                for (int j = 0; j + 1 < nb && n_der > 0; ++j) {
                    const double e = s->lambda_outer_ * s->ell_ * s->ell_ / s->gap_[j];
                    for (int v = 0; v < d; ++v) {
                        const int a = j * d + v, b = (j + 1) * d + v;
                        trip.emplace_back(a, a, e); trip.emplace_back(b, b, e);
                        trip.emplace_back(a, b, -e); trip.emplace_back(b, a, -e);
                    }
                }
                Eigen::SparseMatrix<double> M(nu, nu);
                M.setFromTriplets(trip.begin(), trip.end());
                const matrix_t du = s->inner_sensitivity_(C, M, B_data);
                if (du.size() != 0) {
                    J.topRows(n_obs) += C * du;
                    J.block(n_obs, 0, nu, p) = w.cwiseSqrt().asDiagonal() * du;
                    for (int j = 0; j + 1 < nb && n_der > 0; ++j) {
                        const double c = std::sqrt(s->lambda_outer_ * s->ell_ * s->ell_ / s->gap_[j]);
                        J.block(n_obs + nu + j * d, 0, d, p) =
                          c * (du.block((j + 1) * d, 0, d, p) - du.block(j * d, 0, d, p));
                    }
                }
            }
            if (s->param_map_.active()) {
                const vector_t dm = s->param_map_.chain(theta, vector_t::Ones(p));
                for (int j = 0; j < p; ++j) { J.col(j) *= dm[j]; }
            }
            return J;
        }
    };
    // the L2 half of the penalty per control unknown, and the fitted control flattened to match
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

    struct h1_outer_objective {
        bs_ls_ode_h1* solver;
        double operator()(const vector_t& z) const { return solver->H_(solver->param_map_.theta_of(z)); }
        auto gradient() const {
            return [s = solver](const vector_t& z) {
                const vector_t theta = s->param_map_.theta_of(z);
                vector_t g = s->grad_H_(theta);
                // a NaN gradient would pass BFGS's stopping test (norm > tol is false for NaN) and report
                // the garbage point as converged; zeroing stops the loop on a point the safeguard can discard
                if (!g.allFinite()) { g.setZero(); }
                return s->param_map_.chain(theta, g);
            };
        }
    };

    // the H1 penalty
    double ell_ = 1.0;
    std::vector<double> gap_;
    static constexpr double gap_floor_ = 1e-12;

    std::function<matrix_t(const vector_t&)> ic_jacobian_;
    std::function<vector_t(const vector_t&)> ic_from_theta_;

    vector_t theta_;
    double lambda_outer_ = -1;
    int outer_max_iter_ = 100;
    double outer_tol_ = 1e-6;
    static constexpr double divergent_outer_cost_ = 1e20;
    theta_log_map param_map_;
    bool exact_jacobian_ = true;
    double outer_step_cap_ = 0;

    vector_t best_theta_;
    double best_value_ = std::numeric_limits<double>::infinity();
    bool safeguarded_ = false;

    vector_t theta_cache_;
    bool cache_valid_ = false;
    int inner_failures_ = 0;

    vector_t theta_last_;
    double outer_value_last_ = std::numeric_limits<double>::infinity();
    outer_method outer_method_ = outer_method::bfgs;
    double outer_value_ = 0;
    int outer_n_iter_ = 0;
    bool outer_converged_ = false;
};

}   // namespace internals
}   // namespace fdapde

#endif   // __BS_LS_ODE_H1_SOLVER_H__
