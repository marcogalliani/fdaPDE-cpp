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
[[maybe_unused]] constexpr int OptimizeGCV  = 0x1 << 1;
[[maybe_unused]] constexpr int OptimizeMSRE = 0x2 << 1;
  
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

        int calibration = (flag & 0b11110);   // detect calibration strategy
        n_iter_.clear();
        converged_ = true;
        for (int i = 0; i < rank; ++i) {
            // select optimal smoothing level for i-th component
            std::array<double, n_lambda> opt_lambda;
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
            const auto& [f, s] = solve_(X, opt_lambda, V.col(i), tol_);
            n_iter_.push_back(n_iter_last_);
            converged_ = converged_ && converged_last_;
            for (int j = 0; j < n_lambda; ++j) { lambda_(i, j) = opt_lambda[j]; }
            // store results
            f_norm_[i] = std::sqrt(f.dot(smoother_->mass() * f));
            f_.col(i) = f / f_norm_[i];
            s_.col(i) = s * f_norm_[i];
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
    const std::vector<int>& n_iter() const { return n_iter_; }   // iterations of the final fit(s)
    bool converged() const { return converged_; }                // whether the final fit(s) met the tolerance
   private:
    // finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f)
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, const LambdaT& lambda, const InitT& f0, double tol) {
        // initialization
        vector_t fn = f0;
        vector_t y = X * fn;
        vector_t s(n_units_);
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
            // relative gradient norm of the objective (profiled in f) at s, i.e. the sine of the angle between s and y
            if ((y - s * s.dot(y)).norm() <= tol * y.norm()) {
                converged_last_ = true;
                break;
            }
        }
        return std::make_pair(smoother_->f(), s);
    }
    // fits the rank-1 model at \lambda (on the, possibly deflated, data X) and returns the GCV approximation of its
    // leave-one-location-out prediction error (up to the constant \norm{X}_F^2), with z = X^\top s:
    //   CV(\lambda) = n_locs^2 \norm{z - \Psi f}^2 / (n_locs - Tr[S_\lambda])^2 - \norm{z}^2
    // s depends on \lambda, hence the energy term -\norm{z}^2 cannot be dropped: without it, a large \lambda is
    // rewarded for aligning s with a smooth low-energy direction (GCV collapse on later components)
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, const LambdaT lambda, const InitT& f0) {
        const auto& [f, s] = solve_(X, lambda, f0, std::max(tol_, internals::fpca_calibration_tol));
	std::array<double, n_lambda> lambda_;
	for (int i = 0; i < n_lambda; ++i) { lambda_[i] = lambda[i]; }
        if (edf_map_.find(lambda_) == edf_map_.end()) {   // cache Tr[S]
            edf_map_[lambda_] = smoother_->edf();
        }
        vector_t z = X.transpose() * s;
        double m = n_locs_, dor = m - edf_map_.at(lambda_);
        return m * m * (z - smoother_->Psi() * f).squaredNorm() / (dor * dor) - z.squaredNorm();
    }
    std::unordered_map<std::array<double, n_lambda>, double, internals::std_array_hash<double, n_lambda>> edf_map_;
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level
  
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

        int calibration = (flag & 0b11110);   // detect calibration strategy
        std::array<double, n_lambda> opt_lambda;
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

        int calibration = (flag & 0b11110);   // detect calibration strategy
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
	
        int calibration = (flag & 0b11110);   // detect calibration strategy
        std::array<double, n_lambda> opt_lambda;
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
  
// class for handling nan
template <typename fPCASolver> class fpca_na_impl {
   private:
    using fpca_t = std::decay_t<fPCASolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
   public:
    using smoother_t = typename fPCASolver::smoother_t;
    static constexpr int n_lambda = smoother_t::n_lambda;

    fpca_na_impl() noexcept = default;
    fpca_na_impl(fPCASolver& fpca) noexcept :
        fpca_(std::addressof(fpca)), smoother_(fpca.smoother()), n_dofs_(smoother_->n_dofs()) { }

    template <typename DataT>
    auto fit(const DataT& data, int rank, const std::vector<double>& lambda_grid, int flag) {
        matrix_t X = data.transpose();         // create temporary of mapped data
        binary_t nan_pattern = na_matrix(X);   // compute missingness pattern

        int n_locs_ = X.cols(), n_units_ = X.rows();
        // initialization
        f_.resize(n_dofs_, rank);
        s_.resize(n_units_, rank);
        f_norm_.resize(rank);

        int calibration = (flag & 0b11110);   // detect calibration strategy
        std::vector<double> opt_lambda(n_lambda);
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
        } break;
        case OptimizeMSRE: {
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        }
        // fit with optimal lambda
        const auto& [F, S] = solve_(X, nan_pattern, rank, opt_lambda, flag);
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
   private:
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, const binary_t& nan, int rank, const LambdaT& lambda, int flag) {
        for (int i = 0; i < lambda.size(); ++i) { fdapde_assert(lambda[i] > 0); }
	matrix_t F(n_dofs_ , rank);
	matrix_t S(n_units_, rank);
        matrix_t U  = matrix_t::Zero(n_units_, n_dofs_);
	matrix_t Un = matrix_t::Zero(n_units_, n_dofs_);
	// repeat for increasing rank
        for (int k = 1; k <= rank; ++k) {
            int n_iter = 0;
            double Jold = std::numeric_limits<double>::max(), Jnew = 1.0;
            while (!almost_equal(Jnew, Jold, tol_) && n_iter < max_iter_) {
                // imputation update
                matrix_t Xn = (~nan).select(X, Un);
                Xn.rowwise() -= Xn.colwise().mean();   // re-center
                // rank-k fPCA on imputed data
                const auto& [f, s] = fpca_->fit(X, k, lambda, flag);
                U = s * f.transpose();   // reconstruction update
                // prepare for next iteration
                n_iter++;
                Jold = Jnew;
                Un = U * smoother_->Psi().transpose();
                Jnew = ((~nan).select(X - Un, 0)).squaredNorm() + (U * smoother_->P(lambda) * U.transpose()).trace();
                if (almost_equal(Jnew, Jold, tol_) || n_iter == max_iter_) {
                    F.leftCols(k) = f;
                    S.leftCols(k) = s;
                }
            }
        }
	return std::make_pair(F, S);
    }
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    fpca_t* fpca_;
    const smoother_t* smoother_;
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs

    // MM scheme parameters
    double tol_ = 1e-6;
    int max_iter_ = 100;
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
        discretize(penalty.get().penalty);
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
            // default to OptimMSRE calibration, if no calibration provided
            if (lambda_grid.size() > n_lambda && (flag & 0b11110) == 0) { flag = flag | OptimizeMSRE; }
            internals::fpca_na_impl mm_scheme(solver_);
            const auto& [f, s] = mm_scheme.fit(data_, rank, lambda_grid, flag);
            f_ = std::move(f);
            s_ = std::move(s);
            f_norm_ = solver_.loadings_norm();
        } else {
            // default to OptimGCV calibration, if no calibration provided
            if (lambda_grid.size() > n_lambda && (flag & 0b11110) == 0) { flag = flag | OptimizeGCV; }

            const auto& [f, s] = solver_.fit(data_, rank, lambda_grid, flag);
            f_ = std::move(f);
            s_ = std::move(s);
            f_norm_ = solver_.loadings_norm();
        }
        lambda_ = solver_.lambda();
        // convergence of the iterative solvers (the direct solver is exact)
        if constexpr (requires { solver_.converged(); }) {
            if (!has_nan_) {
                n_iter_ = solver_.n_iter();
                converged_ = solver_.converged();
            }
        }
        return std::tie(f_, s_);
    }
    // observers
    const matrix_t& S() const { return s_; }   // scoring matrix
    const matrix_t& F() const { return f_; }   // loading matrix
    matrix_t Fn() const { return smoother_.Psi() * f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const std::vector<int>& n_iter() const { return n_iter_; }
    bool converged() const { return converged_; }
   private:
    matrix_t data_;         // mapped geoframe data
    smoother_t smoother_;   // variational solver used in the smoothing step
    bool has_nan_;

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated components
    matrix_t lambda_;              // selected level of smoothing for each component
    std::vector<int> n_iter_;      // iterations of the final fit(s) of iterative solvers
    bool converged_ = true;
};

// deduction guide
template <typename GeoFrame, typename Penalty>
fPCA(const std::string& colname, const GeoFrame& gf, Penalty&& solver) -> fPCA<typename Penalty::solver_t>;

}   // namespace fdapde

#endif   // __FPCA_H__
