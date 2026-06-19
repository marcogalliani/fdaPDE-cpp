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

#ifndef __TS_LS_ODE_SOLVER_H__
#define __TS_LS_ODE_SOLVER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// Time-stepping (discretize-then-optimize) least-squares solver for ODE-penalized smoothing.
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

class ts_ls_ode {
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;

    template <typename Penalty> struct is_valid_penalty {
        static constexpr bool value = requires(Penalty penalty) {
            penalty.field();
            penalty.integrator();
            penalty.max_iter();
            penalty.tol();
        };
    };
    template <typename Penalty> static constexpr bool is_valid_penalty_v = is_valid_penalty<Penalty>::value;

   public:
    static constexpr int n_lambda = 1;
    using solver_category = ls_solver;

    ts_ls_ode() noexcept = default;
    // construct from formula + geoframe (time mesh and response read from the frame)
    template <typename GeoFrame, typename Penalty, typename WeightMatrix>
        requires(is_valid_penalty_v<Penalty>)
    ts_ls_ode(const std::string& formula, const GeoFrame& gf, Penalty&& penalty, const WeightMatrix&) {
        discretize(penalty);
        analyze_data(formula, gf);
    }
    template <typename GeoFrame, typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    ts_ls_ode(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty);
        analyze_data(formula, gf);
    }

    // discretize the penalty: store the field, build the integrator, read scheme parameters
    template <typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    void discretize(Penalty&& penalty) {
        field_ = penalty.field();
        integrator_ = penalty.integrator();
        max_iter_ = penalty.max_iter();
        tol_ = penalty.tol();
        // the system dimension d is taken from the response (set in analyze_data), not the field
        if constexpr (requires(Penalty p) { p.has_ic(); }) {
            if (penalty.has_ic()) {
                has_ic_ = true;
                y0_ = penalty.ic();
            }
        }
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
        matrix_t y_obs = gf[0].data().template col<double>(formula_.lhs()).as_matrix();
        fdapde_assert(y_obs.rows() == m_);
        set_response_(y_obs);
        return;
    }

    // main fit entry point: minimize J for the given penalty weight.
    //
    // Reduced (adjoint-method) optimal control formulation: the decision variable is the
    // piecewise-constant control u that drives the prior dynamics y' = f(t, y) + u(t); the
    // trajectory is recovered by forward integration and the gradient of the data misfit w.r.t.
    // u by the consistent discrete adjoint of the RK scheme (RKIntegrator::adjoint_step). The
    // resulting smooth, gradient-based program is solved with BFGS + a backtracking line search.
    const vector_t& fit(double lambda) {
        fdapde_assert(lambda > 0 && d_ > 0 && m_ >= 2 && static_cast<bool>(field_));
        if (has_ic_) { fdapde_assert(y0_.size() == d_); }
        lambda_ = lambda;
        const int nu = (m_ - 1) * d_;
        const bool free_y1 = !has_ic_;
        const int nz = nu + (free_y1 ? d_ : 0);
        // initial decision vector: zero control; initial state from the data-based guess (or the IC)
        vector_t z = vector_t::Zero(nz);
        if (free_y1) { z.segment(nu, d_) = initial_guess_().row(0).transpose(); }
        // minimize the reduced objective over the control with BFGS
        control_objective problem {this};
        BFGS<Dynamic> optimizer(max_iter_, tol_, 1.0);
        // Wolfe line search: its curvature condition keeps the BFGS inverse Hessian positive
        // definite (descent directions), and its Armijo test rejects the divergent-cost surrogate.
        vector_t z_opt = optimizer.optimize(problem, z, WolfeLineSearch());
        // recover trajectory and diagnostics from the optimal control
        Y_ = forward_recover_(z_opt);
        if (has_ic_) { Y_.row(0) = y0_.transpose(); }
        objective_value_ = optimizer.value();
        n_iter_ = optimizer.n_iter();
        converged_ = (n_iter_ < max_iter_);   // stopped on the gradient tolerance, not the iter cap
        compute_control_();
        flatten_();
        return f_;
    }
    template <typename LambdaT>
        requires(internals::is_vector_like_v<LambdaT>)
    const vector_t& fit(LambdaT&& lambda) {
        fdapde_assert(lambda.size() == n_lambda);
        return fit(lambda[0]);
    }

    // TODO: remove gauss-newton hessian computation
    // hutchinson approximation of Tr[S] for the linearized hat matrix S = M H^{-1} M
    double edf(int r = 100, int seed = random_seed) {
        fdapde_assert(m_ > 0 && lambda_ > 0);
        sparse_matrix_t H;
        vector_t g;
        assemble_(Y_, H, g);   // Gauss-Newton Hessian at the current solution
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
    int n_dofs() const { return m_ * d_; }
    int n_obs() const { return n_obs_; }   // number of observed scalar entries
    int n_components() const { return d_; }
    int n_nodes() const { return m_; }
    const vector_t& time_nodes() const { return time_; }
    const vector_t& f() const { return f_; }            // flattened trajectory (m*d), node-major
    const vector_t& fn() const { return f_; }           // fitted at observation nodes (Psi = I)
    vector_t fitted() const { return f_; }
    const vector_t& beta() const { return beta_; }
    const vector_t& misfit() const { return g_; }       // flattened control defects ((m-1)*d)
    const vector_t& response() const { return y_; }     // flattened response (m*d), NaN -> 0
    const matrix_t& trajectory() const { return Y_; }   // m x d
    const matrix_t& control() const { return control_; }
    double objective() const { return objective_value_; }
    int n_iter() const { return n_iter_; }
    bool converged() const { return converged_; }

   private:
    double dt_(int t) const { return time_(t + 1) - time_(t); }

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

    // --- reduced (adjoint-method) optimal control over the control u ---------------------------
    // Decision vector layout: z = [u_0; ...; u_{m-2}; (y_1)], interval-major control blocks of
    // size d, optionally followed by the free initial state y_1 (absent under a hard IC).

    // forward-recover the trajectory: y_1 from z (or the IC), then y_{t+1} = RK step of f + u_t
    matrix_t forward_recover_(const vector_t& z) const {
        const int nu = (m_ - 1) * d_;
        matrix_t Y(m_, d_);
        Y.row(0) = (has_ic_ ? y0_ : vector_t(z.segment(nu, d_))).transpose();
        for (int t = 0; t < m_ - 1; ++t) {
            vector_t u_t = z.segment(t * d_, d_);
            vector_t yc = Y.row(t).transpose();
            // u_t is the (constant) control on interval t; field_.shifted injects it as f + u_t
            Y.row(t + 1) = integrator_.step(field_.shifted(u_t), time_(t), yc, dt_(t)).transpose();
        }
        return Y;
    }

    // control objective J = SSE(observed) + lambda * sum_t dt_t ||u_t||^2
    double objective_(const vector_t& z) const {
        matrix_t Y = forward_recover_(z);
        // a control may drive the (possibly nonlinear) forward integration to blow up: report a
        // large finite cost so the line search backtracks out of the divergent region.
        if (!Y.allFinite()) { return divergent_cost_; }
        double J = 0;
        for (int t = 0; t < m_; ++t) {
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) { J += std::pow(Y(t, v) - y_obs_(t, v), 2); }
            }
        }
        for (int t = 0; t < m_ - 1; ++t) { J += lambda_ * dt_(t) * z.segment(t * d_, d_).squaredNorm(); }
        return J;
    }

    // control gradient dJ/dz by a backward discrete-adjoint sweep (the reduced gradient: the state
    // has been eliminated by forward integration, leaving only the control). Node data-source
    // s_t = 2 * mask_t * (y_t - y^obs_t); the costate satisfies p_t = s_t + (dy_{t+1}/dy_t)^T p_{t+1}
    // with p_m = s_m, while RKIntegrator::adjoint_step yields dJ_data/du_t per interval.
    vector_t gradient_(const vector_t& z) const {
        const int nu = (m_ - 1) * d_;
        const bool free_y1 = !has_ic_;
        matrix_t Y = forward_recover_(z);
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
        vector_t p = source(m_ - 1);
        for (int t = m_ - 2; t >= 0; --t) {
            vector_t u_t = z.segment(t * d_, d_);
            vector_t yc = Y.row(t).transpose();
            auto [p_prop, grad_contrib] =
              integrator_.adjoint_step(field_.shifted(u_t), time_(t), yc, dt_(t), p);
            g.segment(t * d_, d_) = 2.0 * lambda_ * dt_(t) * u_t + grad_contrib;
            p = source(t) + p_prop;
        }
        if (free_y1) { g.segment(nu, d_) = p; }   // dJ/dy_1 = costate at the first node
        return g;
    }

    // objective functor over the control z, adapting J and its gradient to the core BFGS interface
    struct control_objective {
        const ts_ls_ode* solver;
        double operator()(const vector_t& z) const { return solver->objective_(z); }
        auto gradient() const {
            return [s = solver](const vector_t& z) { return s->gradient_(z); };
        }
    };

    // TODO: remove gauss-newton system computation
    // assemble the Gauss-Newton system H * dY = -g for the current trajectory (used by edf())
    void assemble_(const matrix_t& Y, sparse_matrix_t& H, vector_t& g) const {
        const int N = m_ * d_;
        const matrix_t Id = matrix_t::Identity(d_, d_);
        std::vector<Eigen::Triplet<double>> triplets;
        g = vector_t::Zero(N);
        auto is_fixed = [&](int node) { return has_ic_ && node == 0; };
        auto add_block = [&](int rn, int cn, const matrix_t& blk) {
            if (is_fixed(rn) || is_fixed(cn)) { return; }
            for (int r = 0; r < d_; ++r) {
                for (int c = 0; c < d_; ++c) {
                    if (blk(r, c) != 0.0) { triplets.emplace_back(rn * d_ + r, cn * d_ + c, blk(r, c)); }
                }
            }
        };
        // data term
        for (int t = 0; t < m_; ++t) {
            if (is_fixed(t)) { continue; }
            for (int v = 0; v < d_; ++v) {
                if (mask_(t, v) != 0.0) {
                    triplets.emplace_back(t * d_ + v, t * d_ + v, 1.0);
                    g(t * d_ + v) += Y(t, v) - y_obs_(t, v);
                }
            }
        }
        // penalty term: each interval couples nodes t and t+1
        for (int t = 0; t < m_ - 1; ++t) {
            double dt = dt_(t), w = dt, cw = lambda_ * w;
            vector_t yc = Y.row(t).transpose();
            auto [step, flow] = integrator_.step_with_flow_jacobian(field_, time_(t), yc, dt);
            vector_t u = (Y.row(t + 1).transpose() - step) / dt;
            matrix_t Bc = -flow / dt;   // d u_t/d y_t   (d u_t/d y_{t+1} = I/dt)
            add_block(t, t, cw * (Bc.transpose() * Bc));
            add_block(t, t + 1, (cw / dt) * Bc.transpose());
            add_block(t + 1, t, (cw / dt) * Bc);
            add_block(t + 1, t + 1, (cw / (dt * dt)) * Id);
            if (!is_fixed(t)) { g.segment(t * d_, d_) += cw * (Bc.transpose() * u); }
            if (!is_fixed(t + 1)) { g.segment((t + 1) * d_, d_) += (cw / dt) * u; }
        }
        if (has_ic_) {
            for (int v = 0; v < d_; ++v) { triplets.emplace_back(v, v, 1.0); }
        }
        H.resize(N, N);
        H.setFromTriplets(triplets.begin(), triplets.end());
        H.makeCompressed();
        return;
    }

    void compute_control_() {
        control_.resize(m_ - 1, d_);
        for (int t = 0; t < m_ - 1; ++t) {
            double dt = dt_(t);
            vector_t yc = Y_.row(t).transpose();
            vector_t step = integrator_.step(field_, time_(t), yc, dt);
            control_.row(t) = ((Y_.row(t + 1).transpose() - step) / dt).transpose();
        }
        return;
    }
    void flatten_() {
        f_.resize(m_ * d_);
        for (int t = 0; t < m_; ++t) { f_.segment(t * d_, d_) = Y_.row(t).transpose(); }
        g_.resize((m_ - 1) * d_);
        for (int t = 0; t < m_ - 1; ++t) { g_.segment(t * d_, d_) = control_.row(t).transpose(); }
        return;
    }

    // model and discretization
    ode_field field_;
    // type-erased so the solver is independent of the scheme's stage count (see any_rk_integrator)
    any_rk_integrator integrator_;
    int max_iter_ = 50;
    double tol_ = 1e-8;
    bool has_ic_ = false;
    vector_t y0_;
    static constexpr double divergent_cost_ = 1e20;   // objective surrogate for a blown-up forward solve

    // data
    vector_t time_;                    // m time nodes
    int m_ = 0, d_ = 0, n_obs_ = 0;    // n. time instants, n. components, n. observed scalars
    matrix_t y_obs_, y_fill_, mask_;   // observations, NaN-filled copy, observation mask

    // results
    double lambda_ = -1;
    matrix_t Y_, control_;        // fitted trajectory (m x d) and control defects ((m-1) x d)
    vector_t f_, g_, y_, beta_;   // flattened trajectory / control / response
    double objective_value_ = 0;
    int n_iter_ = 0;
    bool converged_ = false;
};

}   // namespace internals

// time-stepping ODE-penalty solver API: penalty descriptor wrapping the vector field, the
// time-integration scheme (Butcher tableau) and an optional initial condition. 
struct ts_ls_ode {
    using solver_t = internals::ts_ls_ode;
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    // the scheme is chosen here (via the ButcherTableau<Stages> constructor argument) and erased
    // into a non-templated any_rk_integrator at construction, so the descriptor, the solver and the
    // model wrapper that consume it are all free of the stage-count template parameter.
    struct penalty_packet {
        ode_field field_;
        any_rk_integrator integrator_;
        vector_t ic_;
        bool has_ic_ = false;
        int max_iter_ = 50;
        double tol_ = 1e-8;
       public:
        penalty_packet(ode_field field, any_rk_integrator integrator, int max_iter, double tol) :
            field_(std::move(field)), integrator_(std::move(integrator)), max_iter_(max_iter), tol_(tol) { }
        penalty_packet(ode_field field, any_rk_integrator integrator, vector_t ic, int max_iter, double tol) :
            field_(std::move(field)), integrator_(std::move(integrator)), ic_(std::move(ic)), has_ic_(true),
            max_iter_(max_iter), tol_(tol) { }
        // observers
        const ode_field& field() const { return field_; }
        const any_rk_integrator& integrator() const { return integrator_; }
        const vector_t& ic() const { return ic_; }
        bool has_ic() const { return has_ic_; }
        int max_iter() const { return max_iter_; }
        double tol() const { return tol_; }
    };
   public:
    // Stages is deduced locally by the constructor (from the tableau) and erased away; the class
    // itself is not templated.
    template <typename Field, int Stages>
    ts_ls_ode(const Field& field, const ButcherTableau<Stages>& tableau, int max_iter = 50, double tol = 1e-8) :
        penalty_(ode_field(field), any_rk_integrator(RKIntegrator(tableau)), max_iter, tol) { }
    template <typename Field, int Stages>
    ts_ls_ode(
      const Field& field, const ButcherTableau<Stages>& tableau, const vector_t& ic, int max_iter = 50,
      double tol = 1e-8) :
        penalty_(ode_field(field), any_rk_integrator(RKIntegrator(tableau)), ic, max_iter, tol) { }
    const penalty_packet& get() const { return penalty_; }
   private:
    penalty_packet penalty_;
};

}   // namespace fdapde

#endif   // __TS_LS_ODE_SOLVER_H__
