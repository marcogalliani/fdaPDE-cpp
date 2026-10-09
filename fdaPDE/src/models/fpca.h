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

#ifndef __FPCA_H__
#define __FPCA_H__

#include "header_check.h"

namespace fdapde {

[[maybe_unused]] constexpr int ComputeRandSVD = 0x1;
[[maybe_unused]] constexpr int ComputeXactSVD = 0x0;

// bits 1 - 5 reserved to calibration strategies
[[maybe_unused]] constexpr int NoCalibration = 0x0;
[[maybe_unused]] constexpr int OptimizeGCV  = 0x1 << 1;
[[maybe_unused]] constexpr int OptimizeMSRE = 0x2 << 1;
[[maybe_unused]] constexpr int CalibrationMask = 0b11110;   // bits 1 - 5

// bit 6: on complete data, estimate a smooth mean and remove it before the fPCA (missing-data fits always estimate it)
[[maybe_unused]] constexpr int ComputeMean = 0x1 << 6;
  
namespace internals {

// the fits used to evaluate the GCV index only need to rank the candidate smoothing levels: they are solved at a
// looser tolerance, while the final fit at the selected smoothing levels uses the solver tolerance
[[maybe_unused]] constexpr double fpca_calibration_tol = 1e-5;

// power iteration based fPCA
// finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f)
template <typename VariationalSolver> class fpca_power_iteration_impl {
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

    struct fit_result {
        vector_t f;
        vector_t s;
        std::vector<double> objective_history;
        int iterations = 0;
        bool monotone = true;
    };
   public:
    using smoother_t = std::decay_t<VariationalSolver>;
    static constexpr int n_lambda = smoother_t::n_lambda;
  
    fpca_power_iteration_impl() noexcept = default;
    fpca_power_iteration_impl(VariationalSolver& smoother) noexcept :
        smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()) { }
    fpca_power_iteration_impl(VariationalSolver& smoother, int max_iter, double tol) noexcept :
        smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()), max_iter_(max_iter), tol_(tol) { }

    template <typename DataT> auto fit(const DataT& data, int rank, const std::vector<double>& lambda_grid, int flag) {
        fdapde_assert(lambda_grid.size() > 0 && lambda_grid.size() % n_lambda == 0);
        matrix_t X = data.transpose();
        n_locs_ = X.cols(), n_units_ = X.rows();
        // first guess of PCs set to a multivariate PCA (SVD)
        matrix_t V;
        if (flag & ComputeRandSVD) {
            RSI<matrix_t> svd(X, rank);
            V = std::move(svd.matrixV());
        } else {
            Eigen::JacobiSVD<matrix_t> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
            V = std::move(svd.matrixV());
        }
        // allocate memory
        f_.resize(n_dofs_, rank);
        s_.resize(n_units_, rank);
        f_norm_.resize(rank);
        lambda_.resize(rank, n_lambda);
        objective_history_.resize(rank);
        iterations_.assign(rank, 0);
        monotone_.assign(rank, true);

        int calibration = (flag & CalibrationMask);   // detect calibration strategy
        n_iter_.clear();
        converged_ = true;
        for (int i = 0; i < rank; ++i) {
            // select optimal smoothing level for i-th component
            Eigen::Matrix<double, n_lambda, 1> opt_lambda;
            switch (calibration) {
            case 0: {   // no calibration
                fdapde_assert(lambda_grid.size() == n_lambda);
                std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
            } break;
            case OptimizeGCV: {
                auto gcv_functor = [&](auto lambda) { return gcv_(X, lambda, V.col(i)); };
                GridSearch<n_lambda> optimizer;
                auto opt_ = optimizer.optimize(gcv_functor, lambda_grid);
                for (int i = 0; i < n_lambda; ++i) { opt_lambda[i] = opt_[i]; }
            } break;
            case OptimizeMSRE: {
            } break;
            default: {
                throw std::runtime_error("Unrecognized calibration option.");
            }
            }
            // fit with optimal lambda
            auto result = solve_(X, opt_lambda, V.col(i), tol_);
            n_iter_.push_back(n_iter_last_);
            converged_ = converged_ && converged_last_;
            for (int j = 0; j < n_lambda; ++j) { lambda_(i, j) = opt_lambda[j]; }
            // store results
            f_norm_[i] = std::sqrt(result.f.dot(smoother_->mass() * result.f));
            f_.col(i) = result.f / f_norm_[i];
            s_.col(i) = result.s * f_norm_[i];
            objective_history_[i] = std::move(result.objective_history);
            iterations_[i] = result.iterations;
            monotone_[i] = result.monotone;
            // deflate
            X = X - s_.col(i) * (smoother_->Psi() * f_.col(i)).transpose();
        }
        return std::tie(f_, s_);
    }
    // observers
    const matrix_t& scores() const { return s_; }
    const matrix_t& loading() const { return f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const smoother_t* smoother() const { return smoother_; }
    const std::vector<std::vector<double>>& objective_history() const { return objective_history_; }
    const std::vector<int>& iterations() const { return iterations_; }
    const std::vector<bool>& monotone() const { return monotone_; }
    const std::vector<int>& n_iter() const { return n_iter_; }   // iterations of the final fit(s)
    bool converged() const { return converged_; }                // whether the final fit(s) met the tolerance
   private:
    // finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f). Stops when the relative gradient norm
    // of the objective drops below tol; the objective J is recorded at each iteration, and flagged if it increases
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, const LambdaT& lambda, const InitT& f0, double tol) {
        // initialization
        vector_t fn = f0;
        vector_t y = X * fn;
        vector_t s(n_units_);
        const double X_norm2 = X.squaredNorm();
        double Jold = std::numeric_limits<double>::max();
        fit_result result;
        result.objective_history.reserve(max_iter_);
        n_iter_last_ = 0;
        converged_last_ = false;
        while (n_iter_last_ < max_iter_) {
            // s = X * fn / \norm(X * fn)
            s = y / y.norm();
            // f = \argmin_f \sum_i (y_i - f(p_i))^2 + \int_D (\Delta f)^2, with y = X^\top * s
            smoother_->update_response(X.transpose() * s);
            smoother_->fit(lambda);
            fn = smoother_->Psi() * smoother_->f();
            y = X * fn;
            n_iter_last_++;
            // J = \norm{X - s * fn^\top}_F^2 + P_{\lambda}(f), expanded using \norm{s} = 1 and y = X * fn
            double Jnew = X_norm2 - 2 * s.dot(y) + fn.squaredNorm() + smoother_->ftPf(lambda);
            result.objective_history.push_back(Jnew);
            if (!std::isfinite(Jnew) || (n_iter_last_ > 1 && (Jnew - Jold) / (1.0 + std::abs(Jold)) > tol)) {
                result.monotone = false;
            }
            Jold = Jnew;
            // relative gradient norm of the objective (profiled in f) at s, i.e. the sine of the angle between s and y
            if ((y - s * s.dot(y)).norm() <= tol * y.norm()) {
                converged_last_ = true;
                break;
            }
        }
        result.iterations = n_iter_last_;
        result.f = smoother_->f();
        result.s = std::move(s);
        return result;
    }
    // fits the rank-1 model at \lambda (on the, possibly deflated, data X) and returns the GCV approximation of its
    // leave-one-location-out prediction error (up to the constant \norm{X}_F^2), with z = X^\top s:
    //   CV(\lambda) = n_locs^2 \norm{z - \Psi f}^2 / (n_locs - Tr[S_\lambda])^2 - \norm{z}^2
    // s depends on \lambda, hence the energy term -\norm{z}^2 cannot be dropped: without it, a large \lambda is
    // rewarded for aligning s with a smooth low-energy direction (GCV collapse on later components)
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, const LambdaT lambda, const InitT& f0) {
        const auto result = solve_(X, lambda, f0, std::max(tol_, internals::fpca_calibration_tol));
        std::array<double, n_lambda> lambda_vec;
        for (int i = 0; i < n_lambda; ++i) { lambda_vec[i] = lambda[i]; }
        if (edf_map_.find(lambda_vec) == edf_map_.end()) {   // cache Tr[S]
            edf_map_[lambda_vec] = smoother_->edf();
        }
        vector_t z = X.transpose() * result.s;
        double m = n_locs_, dor = m - edf_map_.at(lambda_vec);
        return m * m * (z - smoother_->Psi() * result.f).squaredNorm() / (dor * dor) - z.squaredNorm();
    }
    std::unordered_map<std::array<double, n_lambda>, double, internals::std_array_hash<double, n_lambda>> edf_map_;
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level
    std::vector<std::vector<double>> objective_history_;
    std::vector<int> iterations_;
    std::vector<bool> monotone_;
  
    // power iteration algorithm parameters
    double tol_ = 1e-8;     // on the relative gradient norm of the objective
    int max_iter_ = 1000;
    int n_iter_last_ = 0;   // iterations and convergence of the last call to solve_
    bool converged_last_ = true;
    std::vector<int> n_iter_;
    bool converged_ = true;
};

template <typename VariationalSolver> class fpca_subspace_iteration_impl {
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using svd_t    = Eigen::JacobiSVD<matrix_t>;
   public:
    using smoother_t = std::decay_t<VariationalSolver>;
    static constexpr int n_lambda = smoother_t::n_lambda;
    
    fpca_subspace_iteration_impl() noexcept = default;
    fpca_subspace_iteration_impl(VariationalSolver& smoother) noexcept :
        smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()) { }
    fpca_subspace_iteration_impl(VariationalSolver& smoother, int max_iter, double tol) noexcept :
        smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()), max_iter_(max_iter), tol_(tol) { }

    /// @brief jointly estimates the requested components with fixed or GCV-selected penalties
    template <typename DataT> auto fit(const DataT& data, int rank, const std::vector<double>& lambda_grid, int flag) {
        fdapde_assert(lambda_grid.size() > 0 && lambda_grid.size() % n_lambda == 0);
        matrix_t X = data.transpose();
        n_locs_ = X.cols(), n_units_ = X.rows();
        // first guess of PCs set to a multivariate PCA (SVD)
        matrix_t V;
        if (flag & ComputeRandSVD) {
            RSI<matrix_t> svd(X, rank);
            V = std::move(svd.matrixV());
        } else {
            Eigen::JacobiSVD<matrix_t> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
            V = svd.matrixV().leftCols(rank);
        }
        // allocate memory
        f_.resize(n_dofs_, rank);
        s_.resize(n_units_, rank);
        f_norm_.resize(rank);
        lambda_.resize(rank, n_lambda);

        int calibration = (flag & CalibrationMask);   // detect calibration strategy
        Eigen::Matrix<double, n_lambda, 1> opt_lambda;
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
        } break;
        case OptimizeGCV: {
            auto gcv_functor = [&](auto lambda) { return gcv_(X, rank, lambda, V); };
            GridSearch<n_lambda> optimizer;
            auto opt_ = optimizer.optimize(gcv_functor, lambda_grid);
            for (int i = 0; i < n_lambda; ++i) { opt_lambda[i] = opt_[i]; }
        } break;
        case OptimizeMSRE: {
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        }
        // fit with optimal lambda
        auto [F, S] = solve_(X, rank, opt_lambda, V, tol_);
        n_iter_ = {n_iter_last_};
        converged_ = converged_last_;
        for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < n_lambda; ++j) { lambda_(i, j) = opt_lambda[j]; }
        }
	// store results
        for (int i = 0; i < rank; ++i) {
            f_norm_[i] = std::sqrt(F.col(i).dot(smoother_->mass() * F.col(i)));   // L^2 norm
            f_.col(i) = F.col(i) / f_norm_[i];
	    s_.col(i) = S.col(i) * f_norm_[i];
        }
        return std::tie(f_, s_);
    }
    // observers
    const matrix_t& scores() const { return s_; }
    const matrix_t& loading() const { return f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const smoother_t* smoother() const { return smoother_; }
    const std::vector<int>& n_iter() const { return n_iter_; }   // iterations of the final fit(s)
    bool converged() const { return converged_; }                // whether the final fit(s) met the tolerance
  private:
    // finds matrices S, F minimizing \norm{X - S * F^\top}_F^2 + \sum_{i=1}^rank P_{\lambda_i}(f_i)
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, int rank, const LambdaT& lambda, const InitT& F0, double tol) {
        // initialization
        matrix_t Fn = F0;
	matrix_t F(n_dofs_, rank);
        matrix_t S(n_units_, rank);
        matrix_t Y = X * Fn;
        n_iter_last_ = 0;
        converged_last_ = false;
        while (n_iter_last_ < max_iter_) {
            // S = left singular vectors of Y. Spans the same subspace as the orthogonal procrustes solution of
            // \argmin \| X - S * F^\top \|_F^2 subject to S^\top * S = I, and converges to the principal directions,
            // resolving the rotational ambiguity of the common-\lambda solution
            svd_t svd(Y, Eigen::ComputeThinU);
            S = svd.matrixU();
            // f_j = \argmin_f \sum_i (y_i - f_j(p_i))^2 + \int_D (\Delta f_j)^2, with y = X^\top * S_j,
	    // j = 1, ..., rank
            for (int j = 0; j < rank; ++j) {
                smoother_->update_response(X.transpose() * S.col(j));
                smoother_->fit(lambda);
		F .col(j) = smoother_->f();
		Fn.col(j) = smoother_->Psi() * smoother_->f();
            }
            Y = X * Fn;
            n_iter_last_++;
            // relative gradient norm of the objective (profiled in F) at S, on the Stiefel manifold (S^\top Y symmetric)
            if ((Y - S * (S.transpose() * Y)).norm() <= tol * Y.norm()) {
                converged_last_ = true;
                break;
            }
        }
        return std::make_pair(F, S);
    }
    // fits the rank-K model at \lambda and returns the GCV approximation of its leave-one-location-out prediction
    // error (up to the constant \norm{X}_F^2), with Z = X^\top S:
    //   CV(\lambda) = n_locs^2 \norm{Z - \Psi F}_F^2 / (n_locs - Tr[S_\lambda])^2 - \norm{Z}_F^2
    // the energy term -\norm{Z}_F^2 depends on \lambda through S, and penalizes scores drifting toward low-energy
    // directions. CV is invariant to rotations of (S, F)
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, int rank, const LambdaT lambda, const matrix_t F0) {
        const auto& [F, S] = solve_(X, rank, lambda, F0, std::max(tol_, internals::fpca_calibration_tol));
        matrix_t Z = X.transpose() * S;
        double m = n_locs_, dor = m - smoother_->edf(lambda);
        return m * m * (Z - smoother_->Psi() * F).squaredNorm() / (dor * dor) - Z.squaredNorm();
    }

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level

    // subspace iteration algorithm parameters
    double tol_ = 1e-8;     // on the relative gradient norm of the objective
    int max_iter_ = 1000;
    int n_iter_last_ = 0;   // iterations and convergence of the last call to solve_
    bool converged_last_ = true;
    std::vector<int> n_iter_;
    bool converged_ = true;
};

// subspace iteration with per-component smoothing level (\lambda_i) selection.
// Behaves like fpca_subspace_iteration_impl but assigns an independent \lambda to each component.
//
// Calibration. The \lambda-vector minimizes the GCV approximation of the leave-one-location-out prediction error of the
// rank-K fit (S, F) = (S(\lambda), F(\lambda)),
//   CV(\lambda) = \sum_k [ n_locs^2 \norm{z_k - \Psi f_k}^2 / (n_locs - Tr[S_{\lambda_k}])^2 - \norm{z_k}^2 ] + \norm{X}_F^2,
// with z_k = X^\top s_k. The energy terms -\norm{z_k}^2 penalize solutions whose scores drift toward
// low-energy directions (as a weak component smoothed with a very large \lambda_k).
// For fixed S (S^\top S = I) the energy terms are constant and CV is separable: \lambda_k only enters the
// smoothing of z_k. The search exploits this:
//  (0) start from the best common \lambda;
//  (1) propose, for the current S, the per-component minimizers of the separable GCV
//        GCV_k(\lambda) = n_locs * \norm{z_k - \Psi f_k(\lambda)}^2 / (n_locs - Tr[S_\lambda])^2;
//  (2) re-estimate (S, F) at the proposal, and accept it only if CV decreases; otherwise try to move one
//      component at a time. Stop when no move decreases CV.
// CV decreases monotonically, so the result is never worse (in CV) than the best common \lambda.
template <typename VariationalSolver> class fpca_subspace_experimental_impl {
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    struct level_t {        // operators at a smoothing level
        matrix_t B, G;      // \Psi C^{-1} \Psi^\top X^\top and X B
        double edf;         // Tr[S_\lambda]
    };
   public:
    using smoother_t = std::decay_t<VariationalSolver>;
    static constexpr int n_lambda = smoother_t::n_lambda;

    fpca_subspace_experimental_impl() noexcept = default;
    fpca_subspace_experimental_impl(VariationalSolver& smoother) noexcept :
        smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()) { }
    fpca_subspace_experimental_impl(VariationalSolver& smoother, int max_iter, double tol) noexcept :
        smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()), max_iter_(max_iter), tol_(tol) { }

    template <typename DataT> auto fit(const DataT& data, int rank, const std::vector<double>& lambda_grid, int flag) {
        fdapde_assert(lambda_grid.size() > 0 && lambda_grid.size() % n_lambda == 0);
        X_ = data.transpose();
        n_locs_ = X_.cols(), n_units_ = X_.rows();
        levels_.clear();
        // first guess of the scores set to a multivariate PCA (SVD)
        matrix_t S0;
        if (flag & ComputeRandSVD) {
            RSI<matrix_t> svd(X_, rank);
            S0 = svd.matrixU();
        } else {
            Eigen::JacobiSVD<matrix_t> svd(X_, Eigen::ComputeThinU);
            S0 = svd.matrixU().leftCols(rank);
        }
        // allocate memory
        f_.resize(n_dofs_, rank);
        s_.resize(n_units_, rank);
        f_norm_.resize(rank);
        lambda_.resize(rank, n_lambda);
        const int n_points = static_cast<int>(lambda_grid.size()) / n_lambda;
        // smoothing levels of a selection (one grid point index per component)
        auto lambdas_of = [&](const std::vector<int>& sel) {
            matrix_t lambdas(sel.size(), n_lambda);
            for (std::size_t k = 0; k < sel.size(); ++k) {
                for (int j = 0; j < n_lambda; ++j) { lambdas(k, j) = lambda_grid[sel[k] * n_lambda + j]; }
            }
            return lambdas;
        };

        int calibration = (flag & CalibrationMask);   // detect calibration strategy
        matrix_t opt_lambdas(rank, n_lambda);   // selected \lambda, one row per component
        switch (calibration) {
        case 0: {   // no calibration: the single provided \lambda row is shared by all components
            fdapde_assert(lambda_grid.size() == n_lambda);
            opt_lambdas = lambdas_of(std::vector<int>(rank, 0));
        } break;
        case OptimizeGCV: {
            const double m = n_locs_;
            // fit at a selection, rotated to principal directions within groups sharing the same \lambda (as the
            // common \lambda start, whose scores are then used to propose component-specific \lambdas), and its CV
            // score (up to the constant \norm{X}_F^2)
            struct fit_state {
                std::vector<int> sel;
                matrix_t S;
                double cv = std::numeric_limits<double>::max();
            };
            auto evaluate = [&](const std::vector<int>& sel, const matrix_t& S_init) {
                matrix_t lambdas = lambdas_of(sel);
                fit_state state {sel, solve_(lambdas, S_init, std::max(tol_, internals::fpca_calibration_tol)), 0};
                matrix_t Fn = loadings_(lambdas, state.S);
                rotate_(lambdas, state.S, Fn);
                for (int k = 0; k < rank; ++k) {
                    vector_t z = X_.transpose() * state.S.col(k);
                    double dor = m - level_(lambdas.row(k)).edf;
                    state.cv += m * m * (z - Fn.col(k)).squaredNorm() / (dor * dor) - z.squaredNorm();
                }
                return state;
            };
            // (0) best common \lambda
            fit_state best;
            for (int g = 0; g < n_points; ++g) {
                fit_state state = evaluate(std::vector<int>(rank, g), S0);
                if (state.cv < best.cv) { best = std::move(state); }
            }
            const int max_rounds = 10;
            for (int round = 0; round < max_rounds; ++round) {
                // (1) separable proposal for the current scores
                matrix_t Z = X_.transpose() * best.S;
                std::vector<int> proposal(rank, 0);
                std::vector<double> gcv_min(rank, std::numeric_limits<double>::max());
                for (int g = 0; g < n_points; ++g) {
                    const level_t& level = level_(lambdas_of(std::vector<int>(1, g)).row(0));
                    double dor = m - level.edf;
                    for (int k = 0; k < rank; ++k) {
                        double gcv = m * (Z.col(k) - level.B * best.S.col(k)).squaredNorm() / (dor * dor);
                        if (gcv < gcv_min[k]) {
                            gcv_min[k] = gcv;
                            proposal[k] = g;
                        }
                    }
                }
                if (proposal == best.sel) { break; }
                // (2) accept the proposal only if CV decreases at the re-estimated (S, F)
                fit_state state = evaluate(proposal, best.S);
                if (state.cv < best.cv) {
                    best = std::move(state);
                    continue;
                }
                // otherwise, move the single component that decreases CV the most
                fit_state best_move;
                for (int k = 0; k < rank; ++k) {
                    if (proposal[k] == best.sel[k]) { continue; }
                    std::vector<int> sel = best.sel;
                    sel[k] = proposal[k];
                    fit_state move = evaluate(sel, best.S);
                    if (move.cv < best_move.cv) { best_move = std::move(move); }
                }
                if (best_move.cv >= best.cv) { break; }
                best = std::move(best_move);
            }
            opt_lambdas = lambdas_of(best.sel);
            S0 = best.S;
        } break;
        case OptimizeMSRE: {
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        }
        // fit with optimal lambda (warm-started from the selected fit, or from the SVD guess)
        matrix_t S = solve_(opt_lambdas, S0, tol_);
        n_iter_ = {n_iter_last_};
        converged_ = converged_last_;
        matrix_t Fn = loadings_(opt_lambdas, S);
        rotate_(opt_lambdas, S, Fn);
        // expansion coefficients of the loadings: f_k = C_{\lambda_k}^{-1} \Psi^\top X^\top s_k
        matrix_t F(n_dofs_, rank);
        for (int k = 0; k < rank; ++k) {
            smoother_->update_response(X_.transpose() * S.col(k));
            smoother_->fit(opt_lambdas.row(k));
            F.col(k) = smoother_->f();
        }
        // store results
        lambda_ = opt_lambdas;
        for (int i = 0; i < rank; ++i) {
            f_norm_[i] = std::sqrt(F.col(i).dot(smoother_->mass() * F.col(i)));   // L^2 norm
            f_.col(i) = F.col(i) / f_norm_[i];
            s_.col(i) = S.col(i) * f_norm_[i];
        }
        levels_.clear();   // release memory
        return std::tie(f_, s_);
    }
    // observers
    const matrix_t& scores() const { return s_; }
    const matrix_t& loading() const { return f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const smoother_t* smoother() const { return smoother_; }
    const std::vector<int>& n_iter() const { return n_iter_; }   // iterations of the final fit(s)
    bool converged() const { return converged_; }                // whether the final fit(s) met the tolerance
   private:
    // operators at smoothing level lambda (computed on first use: one factorization and n_units solves)
    template <typename LambdaT> const level_t& level_(const LambdaT& lambda) {
        std::array<double, n_lambda> key;
        for (int j = 0; j < n_lambda; ++j) { key[j] = lambda(0, j); }
        auto it = levels_.find(key);
        if (it != levels_.end()) { return it->second; }
        matrix_t lambda_row = lambda;
        level_t level;
        level.B.resize(n_locs_, n_units_);
        for (int i = 0; i < n_units_; ++i) {
            smoother_->update_response(X_.row(i).transpose());
            smoother_->fit(lambda_row.row(0));
            level.B.col(i) = smoother_->Psi() * smoother_->f();
        }
        level.G = X_ * level.B;
        level.edf = smoother_->edf(lambda_row.row(0));   // system already factorized at lambda
        return levels_.emplace(key, std::move(level)).first->second;
    }
    // loadings at locations fitted to the scores: \Psi f_k = B_{\lambda_k} s_k
    matrix_t loadings_(const matrix_t& lambdas, const matrix_t& S) {
        matrix_t Fn(n_locs_, S.cols());
        for (int k = 0; k < S.cols(); ++k) { Fn.col(k) = level_(lambdas.row(k)).B * S.col(k); }
        return Fn;
    }
    // components sharing the same \lambda are identified only up to a rotation. Each such group is rotated so that its
    // block of M = S^\top X \Psi F (symmetric at convergence) is diagonal, with decreasing diagonal (Rayleigh-Ritz step)
    void rotate_(const matrix_t& lambdas, matrix_t& S, matrix_t& Fn) {
        const int rank = S.cols();
        matrix_t M = S.transpose() * X_ * Fn;
        std::vector<bool> done(rank, false);
        for (int i = 0; i < rank; ++i) {
            if (done[i]) { continue; }
            std::vector<int> group;
            for (int j = i; j < rank; ++j) {
                if (!done[j] && lambdas.row(j) == lambdas.row(i)) {
                    group.push_back(j);
                    done[j] = true;
                }
            }
            const int g = group.size();
            if (g == 1) { continue; }
            matrix_t Mg(g, g), Sg(S.rows(), g), Fg(Fn.rows(), g);
            for (int a = 0; a < g; ++a) {
                for (int b = 0; b < g; ++b) { Mg(a, b) = 0.5 * (M(group[a], group[b]) + M(group[b], group[a])); }
                Sg.col(a) = S.col(group[a]);
                Fg.col(a) = Fn.col(group[a]);
            }
            Eigen::SelfAdjointEigenSolver<matrix_t> eig(Mg);
            matrix_t Q = eig.eigenvectors().rowwise().reverse();   // decreasing eigenvalues
            Sg = Sg * Q;
            Fg = Fg * Q;
            for (int a = 0; a < g; ++a) {
                S.col(group[a]) = Sg.col(a);
                Fn.col(group[a]) = Fg.col(a);
            }
        }
    }
    // finds the scores S minimizing \norm{X - S * F^\top}_F^2 + \sum_{j=1}^rank P_{\lambda_j}(f_j) (F profiled out),
    // with an independent \lambda_j (row j of lambdas) for every component
    matrix_t solve_(const matrix_t& lambdas, const matrix_t& S0, double tol) {
        const int rank = S0.cols();
        std::vector<const matrix_t*> G(rank);
        for (int k = 0; k < rank; ++k) { G[k] = &level_(lambdas.row(k)).G; }
        matrix_t S = S0, Y(n_units_, rank);
        n_iter_last_ = 0;
        converged_last_ = false;
        while (true) {
            for (int k = 0; k < rank; ++k) { Y.col(k) = (*G[k]) * S.col(k); }   // X \Psi f_k, with f_k fitted to s_k
            // relative gradient norm of the objective (profiled in F) at S, on the Stiefel manifold
            matrix_t SY = S.transpose() * Y;
            if ((Y - S * (0.5 * (SY + SY.transpose()))).norm() <= tol * Y.norm()) {
                converged_last_ = true;
                break;
            }
            if (n_iter_last_ == max_iter_) { break; }
            // S = \argmin \| X - S * F^\top \|_F^2 subject to S^\top * S = I (orthogonal procrustes problem): S = U * V^\top,
            // being Y = U \Sigma V^\top the thin SVD of Y
            Eigen::JacobiSVD<matrix_t> svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
            S = svd.matrixU() * svd.matrixV().transpose();
            n_iter_last_++;
        }
        return S;
    }
    using key_t = std::array<double, n_lambda>;
    std::unordered_map<key_t, level_t, internals::std_array_hash<double, n_lambda>> levels_;
    matrix_t X_;                   // data (n_units x n_locs)
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level (one row per component)

    // subspace iteration algorithm parameters: iterations are cheap (n_units x n_units products), while the relative
    // rotation of components with distinct \lambda converges slowly, hence the larger default iteration budget
    double tol_ = 1e-8;     // on the relative gradient norm of the objective
    int max_iter_ = 20000;
    int n_iter_last_ = 0;   // iterations and convergence of the last call to solve_
    bool converged_last_ = true;
    std::vector<int> n_iter_;
    bool converged_ = true;
};

// direct fPCA
template <typename VariationalSolver> class fpca_direct_impl {
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
   public:
    using smoother_t = std::decay_t<VariationalSolver>;
    static constexpr int n_lambda = smoother_t::n_lambda;
  
    fpca_direct_impl() noexcept = default;
    fpca_direct_impl(VariationalSolver& smoother) noexcept :
        smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()) { }

    template <typename DataT> auto fit(const DataT& data, int rank, const std::vector<double>& lambda_grid, int flag) {
        fdapde_assert(lambda_grid.size() > 0 && lambda_grid.size() % n_lambda == 0);
        matrix_t X = data.transpose();
        n_locs_ = X.cols(), n_units_ = X.rows();
        // allocate memory
        f_.resize(n_dofs_, rank);
        s_.resize(n_units_, rank);
        f_norm_.resize(rank);
        lambda_.resize(rank, n_lambda);
	
        int calibration = (flag & CalibrationMask);   // detect calibration strategy
        Eigen::Matrix<double, n_lambda, 1> opt_lambda;
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
        } break;
        case OptimizeGCV: {
            auto gcv_functor = [&](auto lambda) { return gcv_(X, rank, lambda, flag); };
            GridSearch<n_lambda> optimizer;
            auto opt_ = optimizer.optimize(gcv_functor, lambda_grid);
            for (int i = 0; i < n_lambda; ++i) { opt_lambda[i] = opt_[i]; }
        } break;
        case OptimizeMSRE: {
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        }
        // fit with optimal lambda
        const auto& [f, s] = solve_(X, rank, opt_lambda, flag);
        // store results
        for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < n_lambda; ++j) { lambda_(i, j) = opt_lambda[j]; }
            f_norm_[i] = std::sqrt(f.col(i).dot(smoother_->mass() * f.col(i)));   // L^2 norm
            f_.col(i) = f.col(i) / f_norm_[i];
	    s_.col(i) = s.col(i) * f_norm_[i];
        }
        return std::tie(f_, s_);
    }
    // observers
    const matrix_t& scores() const { return s_; }
    const matrix_t& loading() const { return f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const smoother_t* smoother() const { return smoother_; }
   private:
    // finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f)
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, int rank, const LambdaT& lambda, int flag) {
        for (int i = 0; i < lambda.size(); ++i) { fdapde_assert(lambda[i] > 0); }
        matrix_t C = smoother_->Psi().transpose() * smoother_->Psi() + smoother_->P(lambda);
        // given the cholesky decomposition of C as C = D * D^\top, compute D^{-1}
        Eigen::LLT<matrix_t> chol(C);
        invD_ = chol.matrixL().solve(matrix_t::Identity(n_dofs_, n_dofs_));
        // compute SVD of X * \Psi * (D^{-1})^\top
        matrix_t V, s;
	vector_t singularValues;
        if (flag & ComputeRandSVD) {
            RSI<matrix_t> svd(X * smoother_->Psi() * invD_.transpose(), rank);
            V = std::move(svd.matrixV());
            singularValues = std::move(svd.singularValues());
	    s = svd.matrixU().leftCols(rank);
        } else {
            Eigen::JacobiSVD<matrix_t> svd(
              X * smoother_->Psi() * invD_.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            V = std::move(svd.matrixV());
            singularValues = std::move(svd.singularValues());
	    s = svd.matrixU().leftCols(rank);
        }
        matrix_t f = (singularValues.head(rank).asDiagonal() * V.leftCols(rank).transpose() * invD_).transpose();
	return std::make_pair(f, s);
    }
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, int rank, const LambdaT lambda, int flag) {
        const auto& [F, S] = solve_(X, rank, lambda, flag);
        // GCV approximation of the leave-one-location-out prediction error (up to \norm{X}_F^2), as in the subspace
        // solver. Tr[S] = Tr[\Psi C^{-1} \Psi^\top] = \|\Psi D^{-\top}\|_F^2, with C = D D^\top (\|D^{-1}\|_F^2 = Tr[C^{-1}]
        // only when \Psi = I, and diverges when \Psi^\top \Psi is rank deficient)
        matrix_t Z = X.transpose() * S;
        double m = n_locs_, dor = m - (smoother_->Psi() * invD_.transpose()).squaredNorm();
        return m * m * (Z - smoother_->Psi() * F).squaredNorm() / (dor * dor) - Z.squaredNorm();
    }
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    matrix_t invD_;                // inverse of the cholesky factor of \Psi^\top * \Psi + P_{\lambda}
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level
};
  
// generates K non-overlapping folds of the observed entries of a partially observed data matrix, for k-fold
// cross-validation of fPCA on missing data
class k_fold_missing_cv_impl {
   private:
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    using fold_t = std::vector<std::pair<int, int>>;

    int n_folds_ = 5;
    std::vector<fold_t> folds_;

    void generate_folds_(const matrix_t& X, int K) {
        std::vector<std::pair<int, int>> duplets;
        // collect all observed (i, j) pairs, shuffle once and deal them to the K folds
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                if (!std::isnan(X(i, j))) duplets.emplace_back(i, j);
            }
        }
        std::shuffle(duplets.begin(), duplets.end(), std::default_random_engine {});
        folds_.resize(K);
        for (std::size_t i = 0; i < duplets.size(); ++i) { folds_[i % K].push_back(duplets[i]); }
    }
   public:
    k_fold_missing_cv_impl(const matrix_t& X, int K) : n_folds_(K) { generate_folds_(X, K); }
    // train and test masks of the fold_index-th split
    std::pair<binary_t, binary_t> split(const matrix_t& X, int fold_index) const {
        binary_t test_mask(X.rows(), X.cols()), train_mask(X.rows(), X.cols());
        for (int k = 0; k < n_folds_; ++k) {
            for (const auto& [i, j] : folds_[k]) {
                if (k == fold_index) {
                    test_mask.set(i, j);
                } else {
                    train_mask.set(i, j);
                }
            }
        }
        return {train_mask, test_mask};
    }
};

// fPCA of partially observed data. MM scheme: impute the missing entries with the current reconstruction, estimate a
// smooth mean and rank-k components on the completed data, repeat until the objective converges. Ranks are fitted
// recursively (k = 1, ..., K, each one warm-started from the previous reconstruction). With a grid of smoothing levels,
// the smoothing level (shared by mean and components) and the rank are selected jointly by GCV or by k-fold
// cross-validation on the observed entries. The k-fold cross-validation runs its (lambda, fold) pairs in parallel (fdaPDE
// execution module, parallel_set_num_threads() threads) when a factory of inner fPCA solvers is given and the smoother
// can detach the factorizations of a copy: each pair on its own copy of the smoother, with its own inner solver. The
// pairs are independent, so the result does not depend on the number of threads
template <typename fPCASolver> class fpca_na_impl {
   private:
    using fpca_t = std::decay_t<fPCASolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
   public:
    using smoother_t = typename fPCASolver::smoother_t;
    static constexpr int n_lambda = smoother_t::n_lambda;

    // makes an inner fPCA solver working on the given smoother (a copy of the smoother of the model)
    using fpca_factory_t = std::function<fpca_t(smoother_t&)>;

    fpca_na_impl() noexcept = default;
    fpca_na_impl(fPCASolver& fpca, smoother_t& smoother) noexcept :
        fpca_(std::addressof(fpca)), smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()) { }
    fpca_na_impl(fPCASolver& fpca, smoother_t& smoother, fpca_factory_t make_fpca) noexcept :
        fpca_(std::addressof(fpca)), smoother_(std::addressof(smoother)), n_dofs_(smoother.n_dofs()),
        make_fpca_(std::move(make_fpca)) { }

    // data is the (n_units x n_locs) data matrix, missing entries are NaN. Returns the mean, loadings and scores
    template <typename DataT>
    auto fit(const DataT& data, int rank, const std::vector<double>& lambda_grid, int flag) {
        fdapde_assert(n_lambda == 1);   // a single smoothing level is supported
        const matrix_t& X = data;
        binary_t nan_pattern = na_matrix(X);   // missingness pattern
        n_locs_ = X.cols(), n_units_ = X.rows();
        int calibration = (flag & CalibrationMask);
        int svd_flag = (flag & ComputeRandSVD);   // the inner fPCA fits only use the initialisation flag
        std::vector<double> opt_lambda_mu(n_lambda);
        std::vector<double> opt_lambda_F(n_lambda);

        matrix_t Un0 = matrix_t::Zero(n_units_, n_locs_);
        Un0.rowwise() += (~nan_pattern).select(X, 0).colwise().mean();   // start by imputing the mean
        int opt_rank;
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda_mu.begin());
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda_F.begin());
            opt_rank = rank;
        } break;
        case OptimizeGCV: {
            gcv_scores_.resize(lambda_grid.size(), rank);
            // the reconstruction is not reset between smoothing levels: each level is warm-started from the rank-K
            // fit of the previous one (the scores may depend on the order of lambda_grid)
            matrix_t Un = Un0;
            for (std::size_t i = 0; i < lambda_grid.size(); ++i) {
                std::vector<double> lambda {lambda_grid[i]};
                for (int k = 1; k <= rank; ++k) {
                    const auto& [mu, F, S, it, conv] =
                      solve_(*smoother_, *fpca_, X, nan_pattern, Un, k, lambda, lambda, svd_flag);
                    Un = S * F.transpose() * smoother_->Psi().transpose();   // reconstruction update
                    Un.rowwise() += mu.transpose() * smoother_->Psi().transpose();
                    double edf = edf_(nan_pattern, mu, S, F, k, lambda, lambda);
                    // the 1.6 factor is an empirical correction of the tendency of GCV to undersmooth
                    double dor = n_units_ * n_locs_ - 1.6 * edf;
                    gcv_scores_(i, k - 1) =
                      n_units_ * n_locs_ / std::pow(dor, 2) * (~nan_pattern).select(X - Un, 0).squaredNorm();
                }
            }
            Eigen::Index opt_lambda_idx, opt_rank_idx;
            gcv_scores_.minCoeff(&opt_lambda_idx, &opt_rank_idx);
            opt_lambda_mu = opt_lambda_F = std::vector<double> {lambda_grid[opt_lambda_idx]};
            opt_rank = opt_rank_idx + 1;   // the rank grid is always {1, ..., rank}
        } break;
        case OptimizeMSRE: {   // k-fold cross-validation on the observed entries
            auto k_fold_cv = k_fold_missing_cv_impl(X, n_folds_);
            const int n_grid = lambda_grid.size(), n_tasks = n_grid * n_folds_;
            // CV error of each (lambda, fold) pair, for each rank: the pairs are independent (each one starts from the
            // mean imputation Un0, the ranks of a pair are fitted in turn)
            matrix_t mse_per_task = matrix_t::Zero(n_tasks, rank);
            auto cv_task = [&](int t, smoother_t& smoother, fpca_t& fpca) {
                const int i = t / n_folds_, j = t % n_folds_;
                std::vector<double> lambda {lambda_grid[i]};
                auto [train_mask, test_mask] = k_fold_cv.split(X, j);
                matrix_t X_train = train_mask.select(X, std::numeric_limits<double>::quiet_NaN());
                binary_t train_nan_pattern = na_matrix(X_train);
                matrix_t Un = Un0;
                for (int k = 1; k <= rank; ++k) {
                    const auto& [mu, F, S, it, conv] =
                      solve_(smoother, fpca, X_train, train_nan_pattern, Un, k, lambda, lambda, svd_flag);
                    Un = S * F.transpose() * smoother.Psi().transpose();   // reconstruction update
                    Un.rowwise() += mu.transpose() * smoother.Psi().transpose();
                    mse_per_task(t, k - 1) = test_mask.select(X - Un, 0).squaredNorm() / test_mask.count();
                }
            };
            constexpr bool detachable = requires(smoother_t& s) { s.detach_factorizations(); };
            if constexpr (detachable) {
                if (make_fpca_ && parallel_get_num_threads() > 1) {
                    parallel_for(0, n_tasks, 1, [&](int t) {   // one task per pair: coarse grained
                        smoother_t smoother = *smoother_;
                        smoother.detach_factorizations();      // its own factorizations, not the ones of smoother_
                        fpca_t fpca = make_fpca_(smoother);
                        cv_task(t, smoother, fpca);
                    });
                } else {
                    for (int t = 0; t < n_tasks; ++t) { cv_task(t, *smoother_, *fpca_); }
                }
            } else {
                for (int t = 0; t < n_tasks; ++t) { cv_task(t, *smoother_, *fpca_); }
            }
            matrix_t mse_table(n_grid, rank);   // CV error for each (lambda, rank), averaged over the folds
            for (int i = 0; i < n_grid; ++i) {
                mse_table.row(i) = mse_per_task.middleRows(i * n_folds_, n_folds_).colwise().mean();
            }
            Eigen::Index opt_lambda_idx, opt_rank_idx;
            mse_table.minCoeff(&opt_lambda_idx, &opt_rank_idx);
            opt_lambda_mu = opt_lambda_F = std::vector<double> {lambda_grid[opt_lambda_idx]};
            opt_rank = opt_rank_idx + 1;   // the rank grid is always {1, ..., rank}
            gcv_scores_ = mse_table.col(opt_rank_idx);   // CV error along the grid, at the selected rank
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        }
        // fit with the selected smoothing level and rank
        center_.resize(n_dofs_);
        f_.resize(n_dofs_, opt_rank);
        s_.resize(n_units_, opt_rank);
        f_norm_.resize(opt_rank);
        lambda_.resize(opt_rank + 1, n_lambda);
        n_iter_.clear();
        converged_ = true;
        matrix_t Un = Un0;
        for (int k = 1; k <= opt_rank; ++k) {
            const auto& [center, F, S, it, conv] =
              solve_(*smoother_, *fpca_, X, nan_pattern, Un, k, opt_lambda_mu, opt_lambda_F, svd_flag);
            Un = S * F.transpose() * smoother_->Psi().transpose();   // reconstruction update
            Un.rowwise() += center.transpose() * smoother_->Psi().transpose();
            center_ = center;
            f_.leftCols(k) = F;
            s_.leftCols(k) = S;
            n_iter_.push_back(it);
            converged_ = converged_ && conv;
        }
        // store results, row 0 of lambda_ is the smoothing level of the mean
        for (int j = 0; j < n_lambda; ++j) { lambda_(0, j) = opt_lambda_mu[j]; }
        for (int i = 0; i < opt_rank; ++i) {
            for (int j = 0; j < n_lambda; ++j) { lambda_(i + 1, j) = opt_lambda_F[j]; }
            f_norm_[i] = std::sqrt(f_.col(i).dot(smoother_->mass() * f_.col(i)));   // L^2 norm
            f_.col(i) = f_.col(i) / f_norm_[i];
            s_.col(i) = s_.col(i) * f_norm_[i];
        }
        return std::make_tuple(center_, f_, s_);
    }
    // observers
    const vector_t& center() const { return center_; }
    const matrix_t& scores() const { return s_; }
    const matrix_t& loading() const { return f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const matrix_t& gcv_scores() const { return gcv_scores_; }
    const std::vector<int>& n_iter() const { return n_iter_; }   // MM iterations of the final fit, for each rank
    bool converged() const { return converged_; }                // whether all final MM fits met the tolerance
   private:
    // result of the MM scheme for one rank
    struct mm_result_t {
        vector_t center;
        matrix_t F, S;
        int n_iter = 0;
        bool converged = false;
    };
    // MM scheme for the rank-k model, started from the (n_units x n_locs) reconstruction Un0. Works on the given smoother
    // and inner fPCA solver only (no member is written): calls on distinct smoother/solver pairs can run concurrently
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    mm_result_t solve_(
      smoother_t& smoother, fpca_t& fpca, const matrix_t& X, const binary_t& nan, const matrix_t& Un0, int rank,
      const LambdaT& lambda_mu, const LambdaT& lambda_F, int flag) const {
        for (int i = 0; i < n_lambda; ++i) {
            fdapde_assert(lambda_mu[i] > 0);
            fdapde_assert(lambda_F[i] > 0);
        }
        mm_result_t r;
        r.center.resize(n_dofs_);
        r.F.resize(n_dofs_, rank);
        r.S.resize(n_units_, rank);
        matrix_t U(n_units_, n_dofs_);
        matrix_t Un = Un0;
        matrix_t Xn(n_units_, n_locs_);
        // penalty matrices of the objective, fixed during the scheme (each P() call solves R0 X = R1)
        const matrix_t P_mu = smoother.P(lambda_mu), P_F = smoother.P(lambda_F);
        double Jold = std::numeric_limits<double>::max(), Jnew = 1.0;
        while (!almost_equal(Jnew, Jold, tol_) && r.n_iter < max_iter_) {
            // imputation update
            Xn = (~nan).select(X, Un);
            // smooth mean of the completed data
            smoother.update_response(Xn.colwise().mean().transpose());
            smoother.fit(lambda_mu[0]);
            vector_t mu = smoother.f();
            Xn.rowwise() -= (smoother.Psi() * mu).transpose();
            // rank-k fPCA of the centred data, at fixed smoothing level
            auto [f, s] = fpca.fit(Xn.transpose(), rank, lambda_F, flag & ComputeRandSVD);
            for (int i = 0; i < rank; ++i) {   // orthonormal scores
                s.col(i) = s.col(i) / fpca.loadings_norm()[i];
                f.col(i) = f.col(i) * fpca.loadings_norm()[i];
            }
            U = s * f.transpose();   // reconstruction update
            U.rowwise() += mu.transpose();
            r.n_iter++;
            Jold = Jnew;
            Un = U * smoother.Psi().transpose();
            Jnew = ((~nan).select(X - Un, 0)).squaredNorm() + n_units_ * mu.transpose() * P_mu * mu +
                   (f.transpose() * P_F * f).trace();
            if (almost_equal(Jnew, Jold, tol_) || r.n_iter == max_iter_) {
                r.center = mu;
                r.F = f;
                r.S = s;
            }
        }
        r.converged = almost_equal(Jnew, Jold, tol_);
        return r;
    }
    // stochastic (Hutchinson) estimate of the effective degrees of freedom of the rank-k fit, single smoothing level
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double edf_(
      const binary_t& nan_pattern, [[maybe_unused]] const vector_t& center, const matrix_t& S,
      [[maybe_unused]] const matrix_t& F, int rank, const LambdaT& lambda_mu, const LambdaT& lambda_F) {
        using triplet_t = Eigen::Triplet<double>;
        std::vector<int> observed_indexes = nan_pattern.which(false);   // row-major ordering
        // B = [1/\sqrt{N} 1_N, S] \kron \Psi, restricted to the observed entries
        matrix_t design_mat(n_units_, rank + 1);
        design_mat.col(0) = 1 / std::sqrt(n_units_) * vector_t::Ones(n_units_);
        design_mat.rightCols(rank) = S;
        sparse_matrix_t B = kronecker(design_mat.sparseView(), smoother_->Psi());
        std::unordered_map<int, int> row_map;   // observed row of B -> row of B_obs
        row_map.reserve(observed_indexes.size());
        for (std::size_t i = 0; i < observed_indexes.size(); ++i) { row_map[observed_indexes[i]] = i; }
        std::vector<triplet_t> B_obs_triplets;
        B_obs_triplets.reserve(B.nonZeros());
        for (int k = 0; k < B.outerSize(); ++k) {
            for (typename sparse_matrix_t::InnerIterator it(B, k); it; ++it) {
                auto map_it = row_map.find(it.row());
                if (map_it != row_map.end()) { B_obs_triplets.emplace_back(map_it->second, it.col(), it.value()); }
            }
        }
        sparse_matrix_t B_obs(observed_indexes.size(), B.cols());
        B_obs.setFromTriplets(B_obs_triplets.begin(), B_obs_triplets.end());
        // smoothing levels: lambda_mu for the mean, lambda_F for the components
        sparse_matrix_t diag_lambdas(rank + 1, rank + 1);
        std::vector<triplet_t> lambda_triplets;
        lambda_triplets.emplace_back(0, 0, lambda_mu[0]);
        for (int i = 0; i < rank; ++i) { lambda_triplets.emplace_back(i + 1, i + 1, lambda_F[0]); }
        diag_lambdas.setFromTriplets(lambda_triplets.begin(), lambda_triplets.end());
        SparseBlockMatrix<double, 2, 2> A(
          B_obs.transpose() * B_obs, kronecker(diag_lambdas, smoother_->stiff()),
          kronecker(diag_lambdas, smoother_->stiff()), kronecker(-diag_lambdas, smoother_->mass()));
        // Tr[S] \approx 1/r \sum_i e_i^\top B A^{-1} B^\top e_i, with e_i Rademacher vectors
        std::mt19937 rng(random_seed);
        rademacher_distribution rademacher;
        matrix_t Us(B.rows(), n_mc_samples_);
        for (int i = 0; i < B.rows(); ++i) {
            for (int j = 0; j < n_mc_samples_; ++j) { Us(i, j) = rademacher(rng); }
        }
        using sparse_solver_t = eigen_sparse_solver_movable_wrap<Eigen::SparseLU<sparse_matrix_t>>;
        sparse_solver_t invA;
        invA.compute(A);
        matrix_t target = matrix_t::Zero(2 * B.cols(), n_mc_samples_);
        target.topRows(B.cols()) = B.transpose() * Us;
        matrix_t y = invA.solve(target);
        double trS = 0.0;
        for (int i = 0; i < n_mc_samples_; ++i) { trS += Us.col(i).dot(B * y.topRows(B.cols()).col(i)); }
        return trS / n_mc_samples_;
    }
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    int n_folds_ = 5;
    int n_mc_samples_ = 100;       // Monte Carlo samples for the trace of the smoothing matrix
    fpca_t* fpca_;
    smoother_t* smoother_;         // smoothing variational solver
    vector_t center_;              // mean expansion coefficient vector
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected smoothing levels, (rank + 1) x n_lambda, row 0 is the mean's
    matrix_t gcv_scores_;          // GCV (#lambda_grid-by-rank) or CV (#lambda_grid-by-1) scores

    // MM scheme parameters
    double tol_ = 1e-4;
    int max_iter_ = 100;
    std::vector<int> n_iter_;
    bool converged_ = true;
    fpca_factory_t make_fpca_ {};  // inner solvers for the parallel k-fold cross-validation (none: sequential)
};

}   // namespace internals

class fpca_power_solver {
    template <typename Smoother> using impl_t = internals::fpca_power_iteration_impl<Smoother>;
   public:
    fpca_power_solver() noexcept : max_iter_(1000), tol_(1e-8) { }
    fpca_power_solver(int max_iter, double tol) noexcept : max_iter_(max_iter), tol_(tol) { }
    template <typename Solver> [[nodiscard]] auto get(Solver&& solver) const {
        return impl_t<Solver>(solver, max_iter_, tol_);
    }
   private:
    int max_iter_;
    double tol_;
};
class fpca_subspace_solver {
    template <typename Smoother> using impl_t = internals::fpca_subspace_iteration_impl<Smoother>;
   public:
    fpca_subspace_solver() noexcept : max_iter_(1000), tol_(1e-8) { }
    fpca_subspace_solver(int max_iter, double tol) noexcept : max_iter_(max_iter), tol_(tol) { }
    template <typename Solver> [[nodiscard]] auto get(Solver&& solver) const {
        return impl_t<Solver>(solver, max_iter_, tol_);
    }
   private:
    int max_iter_;
    double tol_;
};
class fpca_subspace_experimental_solver {
    template <typename Smoother> using impl_t = internals::fpca_subspace_experimental_impl<Smoother>;
   public:
    fpca_subspace_experimental_solver() noexcept : max_iter_(20000), tol_(1e-8) { }
    fpca_subspace_experimental_solver(int max_iter, double tol) noexcept : max_iter_(max_iter), tol_(tol) { }
    template <typename Solver> [[nodiscard]] auto get(Solver&& solver) const {
        return impl_t<Solver>(solver, max_iter_, tol_);
    }
   private:
    int max_iter_;
    double tol_;
};
class fpca_direct_solver {
    template <typename Smoother> using impl_t = internals::fpca_direct_impl<Smoother>;
   public:
    fpca_direct_solver() noexcept = default;
    template <typename Smoother> [[nodiscard]] auto get(Smoother&& solver) const { return impl_t<Smoother>(solver); }
};
  
template <typename VariationalSolver> class fPCA {
   private:
    using smoother_t = std::decay_t<VariationalSolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    static constexpr int n_lambda = smoother_t::n_lambda;
   public:
    fPCA() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    fPCA(const std::string& colname, const GeoFrame& gf, Penalty&& penalty) noexcept : smoother_(), data_() {
        discretize(penalty.get());
        analyze_data(colname, gf);
    }
    template <typename... Args> void discretize(Args&&... args) {
        smoother_.discretize(std::forward<Args>(args)...);
        n_dofs_ = smoother_.n_dofs();
	return;
    }
    template <typename GeoFrame> void analyze_data(const std::string& colname, const GeoFrame& gf) {
        fdapde_assert(gf.n_layers() == 1);
        data_ = gf[0].data().template col<double>(colname).as_matrix();
        n_locs_ = data_.rows();
        n_units_ = data_.cols();
	smoother_.analyze_data(gf, vector_t::Ones(gf[0].rows()).asDiagonal());
        // detect if data_ has at least one missing value
        has_nan_ = false;
        for (int i = 0; i < n_locs_; ++i) {
            for (int j = 0; j < n_units_; ++j) {
                if (std::isnan(data_(i, j))) {
                    has_nan_ = true;
                    break;
                }
            }
        }
	return;
    }

    template <typename LambdaT, typename Policy = fpca_power_solver>
        requires(internals::is_vector_like_v<LambdaT>)
    auto fit(int rank, const LambdaT& lambda_grid, int flag = ComputeRandSVD, Policy policy = Policy()) {
        fdapde_assert(lambda_grid.size() % n_lambda == 0);
        auto solver_ = policy.get(smoother_);   // instantiate solver implementation
        f_.resize(n_dofs_, rank);
        s_.resize(n_units_, rank);
        f_norm_.resize(rank);
        lambda_.resize(n_lambda, rank);
        // dispatch to processing logic
        if (has_nan_) {
            // default to k-fold cross-validation, if no calibration provided
            if (lambda_grid.size() > n_lambda && (flag & CalibrationMask) == 0) { flag = flag | OptimizeMSRE; }
            // the factory makes the inner solvers of the parallel cross-validation, on copies of smoother_
            internals::fpca_na_impl mm_scheme(
              solver_, smoother_, [&policy](smoother_t& smoother) { return policy.get(smoother); });
            const auto& [mu, f, s] = mm_scheme.fit(data_.transpose(), rank, lambda_grid, flag);
            center_ = mu;
            f_ = f;
            s_ = s;
            f_norm_ = mm_scheme.loadings_norm();
            lambda_ = mm_scheme.lambda();
            gcv_scores_ = mm_scheme.gcv_scores();
            objective_history_.assign(f_.cols(), {});
            iterations_.assign(f_.cols(), 0);
            monotone_.assign(f_.cols(), true);
            n_iter_ = mm_scheme.n_iter();
            converged_ = mm_scheme.converged();
            return std::tie(f_, s_);
        }
        // default to GCV calibration, if no calibration provided
        if (lambda_grid.size() > n_lambda && (flag & CalibrationMask) == 0) { flag = flag | OptimizeGCV; }
        matrix_t X = data_;   // n_locs x n_units
        center_ = vector_t::Zero(n_dofs_);
        gcv_scores_.resize(0, 0);
        if (flag & ComputeMean) {
            // smooth mean of the units, smoothing level selected by GCV over lambda_grid
            smoother_.update_response(X.rowwise().mean());
            GridSearch<n_lambda> optimizer;
            auto gcv_functor = [&](auto lambda) {
                smoother_.fit(lambda);
                double dor = n_locs_ - smoother_.edf();   // residual degrees of freedom
                return (n_locs_ / std::pow(dor, 2)) * (smoother_.fn() - smoother_.response()).squaredNorm();
            };
            smoother_.fit(optimizer.optimize(gcv_functor, lambda_grid));
            center_ = smoother_.f();
            X.colwise() -= smoother_.fn();
        }
        const auto& [f, s] = solver_.fit(X, rank, lambda_grid, flag);
        f_ = std::move(f);
        s_ = std::move(s);
        f_norm_ = solver_.loadings_norm();
        lambda_ = solver_.lambda();
        if constexpr (requires {
                          solver_.objective_history();
                          solver_.iterations();
                          solver_.monotone();
                      }) {
            objective_history_ = solver_.objective_history();
            iterations_ = solver_.iterations();
            monotone_ = solver_.monotone();
        } else {
            objective_history_.assign(rank, {});
            iterations_.assign(rank, 0);
            monotone_.assign(rank, true);
        }
        // convergence of the iterative solvers (the direct solver is exact)
        if constexpr (requires { solver_.converged(); }) {
            n_iter_ = solver_.n_iter();
            converged_ = solver_.converged();
        }
        return std::tie(f_, s_);
    }
    // observers
    const matrix_t& S() const { return s_; }   // scoring matrix
    const matrix_t& F() const { return f_; }   // loading matrix
    matrix_t Fn() const { return smoother_.Psi() * f_; }
    const vector_t& center() const { return center_; }                   // mean (zero unless estimated)
    vector_t center_locs() const { return smoother_.Psi() * center_; }   // mean at the data locations
    const matrix_t& gcv_scores() const { return gcv_scores_; }           // calibration scores of missing-data fits
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const std::vector<std::vector<double>>& objective_history() const { return objective_history_; }
    const std::vector<int>& iterations() const { return iterations_; }
    const std::vector<bool>& monotone() const { return monotone_; }
    const std::vector<int>& n_iter() const { return n_iter_; }
    bool converged() const { return converged_; }
   private:
    matrix_t data_;         // mapped geoframe data
    smoother_t smoother_;   // variational solver used in the smoothing step
    bool has_nan_;

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    vector_t center_;              // mean expansion coefficient vector
    std::vector<double> f_norm_;   // L^2 norm of estimated components
    matrix_t lambda_;              // selected level of smoothing for each component
    matrix_t gcv_scores_;          // calibration scores of missing-data fits
    std::vector<std::vector<double>> objective_history_;
    std::vector<int> iterations_;
    std::vector<bool> monotone_;
    std::vector<int> n_iter_;      // iterations of the final fit(s) of iterative solvers
    bool converged_ = true;
};

// deduction guide
template <typename GeoFrame, typename Penalty>
fPCA(const std::string& colname, const GeoFrame& gf, Penalty&& solver) -> fPCA<typename Penalty::solver_t>;

}   // namespace fdapde

#endif   // __FPCA_H__
