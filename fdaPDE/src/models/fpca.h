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
            const auto& [f, s] = solve_(X, opt_lambda, V.col(i));
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
   private:
    // finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f)
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, const LambdaT& lambda, const InitT& f0) {
        // initialization
        vector_t fn = f0;
        vector_t s(n_units_);
        double Jold = std::numeric_limits<double>::max(), Jnew = 1.0;
        int n_iter = 0;
        while (!almost_equal(Jnew, Jold, tol_) && n_iter < max_iter_) {
            // s = X * fn / \norm(X * fn)
            s = X * fn;
            s = s / s.norm();
            // f = \argmin_f \sum_i (y_i - f(p_i))^2 + \int_D (\Delta f)^2, with y = X^\top * s
            smoother_->update_response(X.transpose() * s);
            smoother_->fit(lambda);
            // prepare for next iteration
            n_iter++;
            fn = smoother_->Psi() * smoother_->f();
            Jold = Jnew;
            Jnew = (X - s * fn.transpose()).squaredNorm() + smoother_->ftPf(lambda);
        }
        return std::make_pair(smoother_->f(), s);
    }
    // finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f) and returns the GCV index
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, const LambdaT lambda, const InitT& f0) {
        const auto& [f, s] = solve_(X, lambda, f0);
        // evaluate GCV index at convergence
	std::array<double, n_lambda> lambda_;
	for (int i = 0; i < n_lambda; ++i) { lambda_[i] = lambda[i]; }
        if (edf_map_.find(lambda_) == edf_map_.end()) {   // cache Tr[S]
            edf_map_[lambda_] = smoother_->edf();
        }
        // effective degrees of freedom of the rank-1 layer s * (\Psi * f)^\top: the field smoothing
        // dof Tr[S_\lambda] plus the freely estimated unit-norm score vector s (n_units_ - 1 dof).
        // counting the score estimation prevents the dof from collapsing toward the penalty null
        // space (Tr[S_\lambda] -> 1) as \lambda grows.
        double df = edf_map_.at(lambda_) + (n_units_ - 1);
        double N = static_cast<double>(n_locs_) * n_units_;   // number of fitted entries of X
        double dor = N - df;
        // full reconstruction residual \norm{X - s * (\Psi * f)^\top}_F^2. Unlike the projected
        // residual \norm{\Psi * f - X^\top s}^2, this charges for the energy of X not captured by s,
        // so a high \lambda solution that aligns s with an incidentally smooth but low-energy
        // direction is no longer rewarded (which made the GCV collapse for later components).
        vector_t Pf = smoother_->Psi() * f;
        double rss = (X - s * Pf.transpose()).squaredNorm();
        return (N / (dor * dor)) * rss;
    }
    std::unordered_map<std::array<double, n_lambda>, double, internals::std_array_hash<double, n_lambda>> edf_map_;
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level
  
    // power iteration algorithm parameters
    double tol_ = 1e-6;
    int max_iter_ = 20;
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
	    V = std::move(svd.matrixV());
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
        const auto& [F, S] = solve_(X, rank, opt_lambda, V);
	// store results
        for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < n_lambda; ++j) { lambda_(i, j) = opt_lambda[j]; }
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
  private:
    // finds matrices S, F minimizing \norm{X - S * F^\top}_F^2 + \sum_{i=1}^rank P_{\lambda_i}(f_i)
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, int rank, const LambdaT& lambda, const InitT& F0) {
        // initialization
        matrix_t Fn = F0;
	matrix_t F(n_dofs_, rank);
        matrix_t S(n_units_, rank);
        double Jold = std::numeric_limits<double>::max(), Jnew = 1.0;
        int n_iter = 0;
        while (!almost_equal(Jnew, Jold, tol_) && n_iter < max_iter_) {
            // solve the orthogonal procrustes problem
            // S = \argmin \| X - S * F^\top \|_F^2 subject to S^\top * S = I
            S = X * Fn;
	    svd_t svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
            S = svd.matrixU();
            // f_j = \argmin_f \sum_i (y_i - f_j(p_i))^2 + \int_D (\Delta f_j)^2, with y = X^\top * S_j,
	    // j = 1, ..., rank
            double pen = 0;
            for (int j = 0; j < rank; ++j) {
                smoother_->update_response(X.transpose() * S.col(j));
                smoother_->fit(lambda);
		F .col(j) = smoother_->f();
		Fn.col(j) = smoother_->Psi() * smoother_->f();
		pen = pen + smoother_->ftPf(lambda);
            }
            // prepare for next iteration
            n_iter++;
            Jold = Jnew;
            Jnew = (X - S * Fn.transpose()).squaredNorm() + pen;
        }
        return std::make_pair(F, S);
    }
    // finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f) and returns the GCV index
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, int rank, const LambdaT lambda, const matrix_t F0) {
        const auto& [F, S] = solve_(X, rank, lambda, F0);
        // evaluate GCV index at convergence
        int dor = n_locs_ - smoother_->edf(lambda);
        return (n_locs_ / std::pow(dor, 2)) * (X.transpose() * S - (smoother_->Psi() * F)).squaredNorm();
    }
  
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level

    // subspace iteration algorithm parameters
    double tol_ = 1e-6;
    int max_iter_ = 20;
};

// subspace iteration with per-component smoothing level (\lambda_i) selection.
// Behaves like fpca_subspace_iteration_impl but assigns an independent \lambda to each component,
// chosen *jointly* by cyclic block-coordinate descent on the summed per-component GCV: each
// \lambda_j is re-optimized over the grid with the others held fixed, sweeping until the
// \lambda-vector stops moving. (One sweep would be the conditional/greedy selection; sweeping to
// convergence lets the earlier \lambda's readjust once the later ones are known.)
//
// The GCV index uses the *full* rank-curr reconstruction error \norm{X - S * (\Psi * F)^\top}_F^2.
// Because the score matrix S is orthonormal, this equals the reconstruction error of the deflated
// residual for the component being refined, so it charges for the energy of X not captured by the
// orthonormal score S_curr. This removes the high-\lambda GCV collapse on later components that
// affects the bare projected residual \norm{\Psi f - X^\top s}^2. The effective degrees of freedom
// add the score estimation (n_units_ - curr free parameters, orthogonal to the curr-1 fixed scores)
// to the field smoothing dof Tr[S_\lambda], so they do not collapse toward 1 as \lambda grows.
template <typename VariationalSolver> class fpca_subspace_experimental_impl {
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using svd_t    = Eigen::JacobiSVD<matrix_t>;
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
        matrix_t opt_lambdas_by_pc(rank, n_lambda);   // selected \lambda, one row per component
        switch (calibration) {
        case 0: {   // no calibration: the single provided \lambda row is shared by all components
            fdapde_assert(lambda_grid.size() == n_lambda);
            for (int i = 0; i < rank; ++i) {
                for (int j = 0; j < n_lambda; ++j) { opt_lambdas_by_pc(i, j) = lambda_grid[j]; }
            }
        } break;
        case OptimizeGCV: {
            // Joint per-component \lambda selection by cyclic block-coordinate descent. The joint
            // objective is the sum over components of the same energy-guarded per-component GCV used
            // by the sequential solver, evaluated at the *jointly* estimated orthonormal scores S:
            //   G(\lambda) = \sum_k  N * ||X - S_k (\Psi f_k)^\top||_F^2 / (N - Tr[S_{\lambda_k}] - (n_units-1))^2,
            //   with N = n_locs * n_units.
            // The components are coupled through the shared frame S, so each \lambda_k is grid-searched
            // with the others held fixed, sweeping until the \lambda-vector stops moving: one sweep is
            // the conditional/greedy selection, sweeping to convergence yields a coordinate-wise minimum
            // of the joint G. Cost is O(sweeps * rank * |grid|) -- linear in rank (unlike a simplex search).
            //
            // The full-reconstruction residual ||X - S_k (\Psi f_k)^\top||^2 = ||X||^2 - ||X^\top S_k||^2 +
            // ||\Psi f_k - X^\top S_k||^2 carries the energy guard -||X^\top S_k||^2. This is essential:
            // orthogonality of S does NOT prevent the high-\lambda collapse, because the joint Procrustes
            // step can still rotate S_k (within the complement of the other scores) toward a low-energy
            // smooth direction at large \lambda. The guard charges for that lost energy, so weak
            // components keep an interior optimum instead of latching onto the over-smoothing end.
            const int n_points = static_cast<int>(lambda_grid.size()) / n_lambda;
            const double N = static_cast<double>(n_locs_) * n_units_;   // number of fitted entries of X
            // sum of per-component energy-guarded GCV at smoothing matrix `lambdas`, solved from `F0`;
            // the converged loadings (evaluated at the locations) are returned in `Fn_out` for warm-starts
            auto full_gcv = [&](const matrix_t& lambdas, const matrix_t& F0, matrix_t& Fn_out) -> double {
                const auto& [F, S] = solve_(X, rank, lambdas, F0);
                Fn_out = smoother_->Psi() * F;   // n_locs x rank
                double g = 0;
                for (int k = 0; k < rank; ++k) {
                    std::array<double, n_lambda> key;
                    for (int m = 0; m < n_lambda; ++m) { key[m] = lambdas(k, m); }
                    if (edf_map_.find(key) == edf_map_.end()) { edf_map_[key] = smoother_->edf(lambdas.row(k)); }
                    double dor = N - (edf_map_.at(key) + (n_units_ - 1));   // field dof + score dof
                    // full-X reconstruction of the k-th rank-1 layer S_k (\Psi f_k)^T (energy-guarded)
                    double rss = (X - S.col(k) * Fn_out.col(k).transpose()).squaredNorm();
                    g += (N / (dor * dor)) * rss;
                }
                return g;
            };
            // start every component at the first grid point
            std::vector<int> sel(rank, 0);
            for (int j = 0; j < rank; ++j) {
                for (int m = 0; m < n_lambda; ++m) { opt_lambdas_by_pc(j, m) = lambda_grid[m]; }
            }
            matrix_t Fn_warm = V;   // warm-start frame, carried across coordinate searches
            // coordinate-descent sweeps (capped to avoid oscillation between near-tied grid points)
            const int max_sweeps = 10;
            for (int sweep = 0; sweep < max_sweeps; ++sweep) {
                bool changed = false;
                for (int j = 0; j < rank; ++j) {
                    double best = std::numeric_limits<double>::max();
                    int best_g = sel[j];
                    matrix_t Fn_best = Fn_warm;
                    // 1-D grid search for component j, the other R-1 \lambda's held fixed. The same
                    // warm frame seeds every candidate, so the GCV values are compared consistently.
                    for (int g = 0; g < n_points; ++g) {
                        for (int m = 0; m < n_lambda; ++m) { opt_lambdas_by_pc(j, m) = lambda_grid[g * n_lambda + m]; }
                        matrix_t Fn_tmp;
                        double v = full_gcv(opt_lambdas_by_pc, Fn_warm, Fn_tmp);
                        if (v < best) { best = v; best_g = g; Fn_best = std::move(Fn_tmp); }
                    }
                    if (best_g != sel[j]) { sel[j] = best_g; changed = true; }
                    for (int m = 0; m < n_lambda; ++m) { opt_lambdas_by_pc(j, m) = lambda_grid[best_g * n_lambda + m]; }
                    Fn_warm = std::move(Fn_best);   // carry the selected frame into the next coordinate
                }
                if (!changed) { break; }   // a full sweep with no change -> coordinate-wise optimum
            }
        } break;
        case OptimizeMSRE: {
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        }
        // fit with optimal lambda (warm-started from the SVD guess)
        const auto& [F, S] = solve_(X, rank, opt_lambdas_by_pc, V);
        // store results
        lambda_ = opt_lambdas_by_pc;
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
   private:
    // finds matrices S, F minimizing \norm{X - S * F^\top}_F^2 + \sum_{j=1}^rank P_{\lambda_j}(f_j),
    // with an independent \lambda_j (row j of lambda) for every component
    template <typename LambdaT, typename InitT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, int rank, const LambdaT& lambda, const InitT& F0) {
        // initialization
        matrix_t Fn = F0;
        matrix_t F(n_dofs_, rank);
        matrix_t S(n_units_, rank);
        double Jold = std::numeric_limits<double>::max(), Jnew = 1.0;
        int n_iter = 0;
        while (!almost_equal(Jnew, Jold, tol_) && n_iter < max_iter_) {
            // solve the orthogonal procrustes problem
            // S = \argmin \| X - S * F^\top \|_F^2 subject to S^\top * S = I
            S = X * Fn;
            svd_t svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
            S = svd.matrixU();
            // f_j = \argmin_f \sum_i (y_i - f_j(p_i))^2 + \lambda_j \int_D (\Delta f_j)^2, y = X^\top S_j
            double pen = 0;
            for (int j = 0; j < rank; ++j) {
                smoother_->update_response(X.transpose() * S.col(j));
                smoother_->fit(lambda.row(j));
                F .col(j) = smoother_->f();
                Fn.col(j) = smoother_->Psi() * smoother_->f();
                pen = pen + smoother_->ftPf(lambda.row(j));
            }
            // prepare for next iteration
            n_iter++;
            Jold = Jnew;
            Jnew = (X - S * Fn.transpose()).squaredNorm() + pen;
        }
        return std::make_pair(F, S);
    }
    std::unordered_map<std::array<double, n_lambda>, double, internals::std_array_hash<double, n_lambda>> edf_map_;
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    smoother_t* smoother_;         // smoothing variational solver
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level (one row per component)

    // subspace iteration algorithm parameters
    double tol_ = 1e-6;
    int max_iter_ = 20;
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
        // evaluate GCV index at convergence (note that Tr[S] = \|D^(-1)\|_F^2)
        int dor = n_locs_ - invD_.squaredNorm();
        return (n_locs_ / std::pow(dor, 2)) * (X.transpose() * S - (smoother_->Psi() * F)).squaredNorm();
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
    fpca_power_solver() noexcept : max_iter_(20), tol_(1e-6) { }
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
    fpca_subspace_solver() noexcept : max_iter_(20), tol_(1e-6) { }
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
    fpca_subspace_experimental_solver() noexcept : max_iter_(20), tol_(1e-6) { }
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
        return std::tie(f_, s_);
    }
    // observers
    const matrix_t& S() const { return s_; }   // scoring matrix
    const matrix_t& F() const { return f_; }   // loading matrix
    matrix_t Fn() const { return smoother_.Psi() * f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
   private:
    matrix_t data_;         // mapped geoframe data
    smoother_t smoother_;   // variational solver used in the smoothing step
    bool has_nan_;

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated components
    matrix_t lambda_;              // selected level of smoothing for each component
};

// deduction guide
template <typename GeoFrame, typename Penalty>
fPCA(const std::string& colname, const GeoFrame& gf, Penalty&& solver) -> fPCA<typename Penalty::solver_t>;

}   // namespace fdapde

#endif   // __FPCA_H__
