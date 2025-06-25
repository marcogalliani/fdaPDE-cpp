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
                // GridOptimizer<n_lambda> optimizer;
                GridOptimizer<n_lambda> optimizer;
                opt_lambda = optimizer.optimize(gcv_functor, lambda_grid);
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
        if (edf_map_.find(lambda) == edf_map_.end()) {   // cache Tr[S]
            edf_map_[lambda] = smoother_->edf();
        }
        int dor = n_locs_ - edf_map_.at(lambda);
        return (n_locs_ / std::pow(dor, 2)) * ((smoother_->Psi() * f) - smoother_->response()).squaredNorm();
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
            // GridOptimizer<n_lambda> optimizer;
            GridOptimizer<n_lambda> optimizer;
            opt_lambda = optimizer.optimize(gcv_functor, lambda_grid);
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
            for (int j = 0; j < n_lambda; ++j) {
                lambda_(i, j) = opt_lambda[j];
            }
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
            // GridOptimizer<n_lambda> optimizer;
            GridOptimizer<n_lambda> optimizer;
            opt_lambda = optimizer.optimize(gcv_functor, lambda_grid);
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
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
   public:
    using smoother_t = typename fPCASolver::smoother_t;
    static constexpr int n_lambda = smoother_t::n_lambda;

    fpca_na_impl() noexcept = default;
    fpca_na_impl(fPCASolver& fpca) noexcept :
        fpca_(std::addressof(fpca)), smoother_(*fpca.smoother()), n_dofs_(fpca.smoother()->n_dofs()) { }

    template <typename DataT>
    auto fit(const DataT& data, int rank, const std::vector<double>& lambda_grid, int flag) {
        matrix_t X = data.transpose();         // create temporary of mapped data
        binary_t nan_pattern = na_matrix(X);   // compute missingness pattern
        n_locs_ = X.cols(), n_units_ = X.rows();

        int calibration = (flag & 0b11110);   // detect calibration strategy
        std::vector<double> opt_lambda(n_lambda);
        int opt_rank;
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
            opt_rank = rank;
        } break;
        case OptimizeGCV: {
            //GCV computation
            //-> Trace estimation
            //we need to vectorize X by stacking its rows (rather than its cols, see the derivation of this GCV)
            //construct the matrix of observed pairs (unit,location): B = D_w(I_N kron_prod Psi)
            //(1) keep track of the indexes of the available observations (.which() already returns the indices in row-major order)
            std::vector<int> observed_indexes = nan_pattern.which(true);
            //(2) create B setting to zero the rows of the kronecker product (I_N kron_prod Psi) corresponding to unobserved locations
            using triplet_t = Eigen::Triplet<double>;
            std::vector<triplet_t> triplets;
            for (int i = 0; i < observed_indexes.size(); ++i) {
                triplets.emplace_back(observed_indexes[i], observed_indexes[i], 1.0); // R[i, selected_rows[i]] = 1
            }
            sparse_matrix_t row_selector(n_units_*n_locs_, n_units_*n_locs_);
            row_selector.setFromTriplets(triplets.begin(), triplets.end());
            sparse_matrix_t Id_N(n_units_,n_units_);
            Id_N.setIdentity();
            sparse_matrix_t B = row_selector*kronecker(Id_N, smoother_.Psi());
            //define the gcv functor
            auto gcv_functor = [&](auto lambda) { return gcv_(X, nan_pattern, B, rank, lambda, flag & 0x1); };
            GridOptimizer<Dynamic> optimizer(n_lambda);
            opt_lambda = optimizer.optimize(gcv_functor, lambda_grid);
            opt_rank = rank;
        } break;
        case OptimizeMSRE: {
            //Utilities to generate non-overlapping folds
            using IndexList = std::vector<std::pair<int, int>>;
            using TrainTestPartition = std::pair<BinaryMatrix<Dynamic>, BinaryMatrix<Dynamic>>;
            // Function to generate K non-overlapping folds
            auto generate_folds = [&](const matrix_t& X, int K) -> std::vector<IndexList> {
                int n = X.rows(), m = X.cols();
                std::vector<std::pair<int, int>> duplets;
                // collect all observed (i, j) pairs
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < m; ++j)
                        if (!std::isnan(X(i, j))) duplets.emplace_back(i, j);
                // shuffle once
                std::shuffle(duplets.begin(), duplets.end(), std::default_random_engine{});
                // partition into K folds
                std::vector<IndexList> folds(K);
                for (std::size_t i = 0; i < duplets.size(); ++i)
                    folds[i % K].push_back(duplets[i]);
                return folds;
            };
            // Split function using precomputed folds
            auto split = [&](const matrix_t& X,
                             const std::vector<IndexList>& folds, int fold_index) -> TrainTestPartition {
                int n = X.rows(), m = X.cols();
                BinaryMatrix<Dynamic> test_mask(n, m), train_mask(n, m);
                for (int k = 0; k < folds.size(); ++k) {
                    for (const auto& [i, j] : folds[k]) {
                        if (k == fold_index)
                            test_mask.set(i, j);
                        else
                            train_mask.set(i, j);
                    }
                }
                return {train_mask, test_mask};
            };
            std::vector<IndexList> folds = generate_folds(X, n_folds_);
            // store the rmse estimated via KCV for each pair (lambda,rank)
            matrix_t rmse_table = matrix_t::Zero(lambda_grid.size(),rank);
            for(int i = 0; i < lambda_grid.size(); ++i) {
                matrix_t mse_per_fold = matrix_t::Zero(n_folds_, rank);
                for(int j = 0; j < n_folds_; ++j) {
                    auto [train_mask, test_mask] = split(X, folds, j);
                    matrix_t X_train = train_mask.select(X,std::numeric_limits<double>::quiet_NaN());
                    binary_t train_nan_pattern = na_matrix(X_train);
                    // repeat for increasing rank
                    matrix_t U = matrix_t::Zero(n_units_, n_dofs_); //starting guess
                    for (int k = 1; k <= rank; ++k) {
                        const auto& [mu, F, S] = solve_(X_train, train_nan_pattern, U,k, std::vector<double>{lambda_grid[i]}, flag & 0x1);
                        U = S * F.transpose();
                        U.rowwise() += mu.transpose(); // reconstruction update
                        // check the loss for each k in {1,...,rank}
                        double loss = test_mask.select(X-U*smoother_.Psi().transpose(),0).squaredNorm();
                        double n_test_obs = test_mask.count();
                        //average for the number of observations
                        mse_per_fold(j,k-1) = loss/n_test_obs;
                    } // rank-recursion
                } //iteration over the folds
                rmse_table.row(i) = mse_per_fold.colwise().mean();
            } // iteration over the lambda
            std::cout << rmse_table << std::endl;
            rmse_table = rmse_table.array() / n_folds_;
            Eigen::Index opt_lambda_idx, opt_rank_idx;
            rmse_table.minCoeff(&opt_lambda_idx, &opt_rank_idx);
            opt_lambda = std::vector<double>{lambda_grid[opt_lambda_idx]};
            opt_rank = opt_rank_idx + 1; //rank grid is always {1,...,rank}
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        }
        // initialization
        center_.resize(n_dofs_);
        f_.resize(n_dofs_, opt_rank);
        s_.resize(n_units_, opt_rank);
        f_norm_.resize(opt_rank);
        lambda_.resize(opt_rank, n_lambda);
        // fit with optimal lambda
        matrix_t U = matrix_t::Zero(n_units_, n_dofs_); //starting guess
        for (int k = opt_rank; k <= opt_rank; ++k) {
            const auto& [center, F, S] = solve_(X, nan_pattern, U,k, opt_lambda, flag & 0x1);
            U = S * F.transpose(); // reconstruction update
            U.rowwise() += center.transpose();
            center_ = center;
            f_.leftCols(k) = F;
            s_.leftCols(k) = S;
        }
        // store results
        for (int i = 0; i < opt_rank; ++i) {
            for (int j = 0; j < n_lambda; ++j) { lambda_(i, j) = opt_lambda[j]; }
            f_norm_[i] = std::sqrt(f_.col(i).dot(smoother_.mass() * f_.col(i)));   // L^2 norm
            f_.col(i) = f_.col(i) / f_norm_[i];
            s_.col(i) = s_.col(i) * f_norm_[i];
        }
        return std::make_tuple(center_, f_, s_);
    }
    // observers
    const matrix_t& scores() const { return s_; }
    const matrix_t& loading() const { return f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
   private:
    // the solve_ method implements the MM scheme (the rank recursion is performed during the fit)
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, const binary_t& nan, const matrix_t& U0, int rank, const LambdaT& lambda, int flag) {
        for (int i = 0; i < lambda.size(); ++i) { fdapde_assert(lambda[i] > 0); }
        vector_t center(n_dofs_);
        matrix_t F(n_dofs_ , rank);
	    matrix_t S(n_units_, rank);
        matrix_t U = U0;
	    matrix_t Un = U * smoother_.Psi().transpose();
        // iterate until the MM scheme converges
        int n_iter = 0;
        double Jold = std::numeric_limits<double>::max(), Jnew = 1.0;
        while (!almost_equal(Jnew, Jold, tol_) && n_iter < max_iter_) {
            // imputation update
            matrix_t Xn = (~nan).select(X, Un);
            //Xn.rowwise() -= Xn.colwise().mean();   // re-center

            // Alternative impl.
            // compute the smooth mean
            smoother_.update_response(Xn.colwise().mean());
            smoother_.fit(lambda);
            const auto mu = smoother_.f();
            Xn.rowwise() -= (smoother_.Psi() * mu).transpose();

            // rank-k fPCA on imputed data
            const auto [f, s] = fpca_->fit(Xn.transpose(), rank, lambda, flag & 0x1); //always run with no calibration
            U = s * f.transpose();   // reconstruction update
            U.rowwise() += mu.transpose();
            // prepare for next iteration
            n_iter++;
            Jold = Jnew;
            Un = U * smoother_.Psi().transpose();
            Jnew = ((~nan).select(X - Un, 0)).squaredNorm() + (U * smoother_.P(lambda) * U.transpose()).trace();
            if (almost_equal(Jnew, Jold, tol_) || n_iter == max_iter_) {
                center = mu; F = f; S = s;
            }
        }
        if (n_iter>=max_iter_){ std::cout << "convergence issues" << std::endl;}
	    return std::make_tuple(center, F, S);
    }
    // finds vectors s, f minimizing \norm{X - s * f^\top}_F^2 + P_{\lambda}(f) and returns the GCV index
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, const binary_t& nan_pattern, const sparse_matrix_t& B, int rank, const LambdaT lambda, int flag) {
        int r = 100;
        sparse_matrix_t Id(n_units_,n_units_);
        Id.setIdentity();
        // sparse_matrix_t IdKronP = kronecker(Id,sparse_matrix_t(smoother_.P(lambda).sparseView()));
        SparseBlockMatrix<double, 2, 2> A(
               B.transpose()*B,                                     lambda[0] * kronecker(Id,smoother_.stiff()),
               lambda[0] * kronecker(Id,smoother_.stiff()),     -lambda[0] * kronecker(Id,smoother_.mass()));
        //construct the matrix to be inverted: B^TB + lambda(I_N kron_prod P)
        // sparse_matrix_t smoothing_mat = B.transpose()*B + IdKronP;
        //sample from the rademacher distribution
        std::mt19937 rng(random_seed);
        rademacher_distribution rademacher;
        matrix_t Us(B.rows(), r);
        for (int i = 0; i < B.rows(); ++i) {
            for (int j = 0; j < r; ++j) { Us(i, j) = rademacher(rng); }
        }
        //solve the system: (B^TB + lambda(I_N kron_prod P))y = B^T e (where e is a sampled vector)
        using sparse_solver_t = eigen_sparse_solver_movable_wrap<Eigen::SparseLU<sparse_matrix_t>>;
        sparse_solver_t invA;
        //using sparse block matrix
        invA.compute(A); // invA.compute(smoothing_mat); //not so sparse
        matrix_t target = matrix_t::Zero(2*B.cols(),r);
        target.topRows(B.cols()) = B.transpose()*Us;
        matrix_t y = invA.solve(target); // matrix_t x = invA.solve(B.transpose()*Us);
        //-> Trace estimation
        //compute e^T*B*y and estimate the trace by averaging these values
        double trS = 0.0;   // monte carlo Tr[S] approximation
        for (int i = 0; i < r; ++i) { trS += Us.col(i).dot(B*y.topRows(B.cols()).col(i)); }
        trS =  trS / r;
        //-> GCV
        //(1) compute the data-fidelity term by fitting the model with the considered lambda
        matrix_t U = matrix_t::Zero(n_units_, n_dofs_); //starting guess
        for (int k = rank; k <= rank; ++k) {
            const auto& [center, f, s] = solve_(X, nan_pattern, U,k, lambda, flag & 0x1);
            U = s * f.transpose(); // reconstruction update
            U.rowwise() += center.transpose();
        }
        //(2) compute the GCV using the estimated dof of the model given by the estimated trace of smoothing matrix
        double dor = B.rows() - trS;
        double gcv  = B.rows() / std::pow(dor, 2) * (~nan_pattern).select(X - U*smoother_.Psi().transpose(), 0).squaredNorm();
        return gcv;
    }

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    int n_folds_ = 5;
    fpca_t* fpca_;
    smoother_t smoother_;
    vector_t center_;              // mean expansion coefficient vector
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level

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
    using data_t   = Eigen::Map<const Eigen::Matrix<double, Dynamic, Dynamic, Eigen::ColMajor>>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    static constexpr int n_lambda = smoother_t::n_lambda;
   public:
    fPCA() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    fPCA(const std::string& colname, const GeoFrame& gf, Penalty&& penalty) noexcept :
        smoother_(), data_(gf[0].data().template col<double>(colname).as_matrix()) {
        fdapde_assert(gf.n_layers() == 1);
        n_locs_ = data_.rows();
	    n_units_ = data_.cols();
        if constexpr (requires(Penalty p) { p.get(); }) {
            smoother_ = smoother_t(gf, penalty.get());
        } else {
            smoother_ = smoother_t(gf, penalty(gf.template triangulation<0>()).get());
        }
	    n_dofs_ = smoother_.n_dofs();
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
    }

    template <typename LambdaT, typename Policy = fpca_power_solver>
        requires(internals::is_vector_like_v<LambdaT>)
    auto fit(int rank, const LambdaT& lambda_grid, int flag = ComputeRandSVD, Policy policy = Policy()) {
        fdapde_assert(lambda_grid.size() % n_lambda == 0);
        auto solver_ = policy.get(smoother_);   // instantiate solver implementation
        f_.resize(n_dofs_ , rank);
        s_.resize(n_units_, rank);
        f_norm_.resize(rank);
        lambda_.resize(n_lambda, rank);
	    // dispatch to processing logic
        if (has_nan_) {
            // default to OptimMSRE calibration, if no calibration provided
            if (lambda_grid.size() > n_lambda && (flag & 0b11110) == 0) { flag = flag | OptimizeMSRE; }
            internals::fpca_na_impl mm_scheme(solver_);
            const auto& [mu,f, s] = mm_scheme.fit(data_, rank, lambda_grid, flag);
            center_ = std::move(mu);
            f_ = std::move(f);
            s_ = std::move(s);
            f_norm_ = solver_.loadings_norm();
            lambda_ = solver_.lambda();
        } else {
            // default to OptimGCV calibration, if no calibration provided
            if (lambda_grid.size() > n_lambda && (flag & 0b11110) == 0) { flag = flag | OptimizeGCV; }
            // data centering
            matrix_t centred_data = data_.transpose();
            smoother_.update_response(centred_data.colwise().mean());
            GridOptimizer<1> opt;
            auto gcv_functor = [&](auto lambda) {
                smoother_.fit(lambda);
                double dor =  n_locs_ - smoother_.edf();  // residual degrees of freedom
                return (n_locs_ / std::pow(dor, 2)) * (~smoother_.nan_pattern()).select(smoother_.fn() - smoother_.response(),0).squaredNorm();
            };
            opt.optimize(gcv_functor, lambda_grid);
            smoother_.fit(opt.optimum());
            center_ = std::move(smoother_.f());
            centred_data.rowwise() -= smoother_.fn().transpose();

            // performing fPCA on the zero-centred data
            const auto& [f, s] = solver_.fit(centred_data.transpose(), rank, lambda_grid, flag);
            f_ = std::move(f);
            s_ = std::move(s);
            f_norm_ = solver_.loadings_norm();
            lambda_ = solver_.lambda();
        }
        return std::tie(f_, s_);
    }
    // observers
    const vector_t& center() const { return center_;} // mean vector
    const matrix_t& S() const { return s_; }   // scoring matrix
    const matrix_t& F() const { return f_; }   // loading matrix
    matrix_t Fn() const { return smoother_.Psi() * f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
   private:
    data_t data_;           // mapped geoframe data
    smoother_t smoother_;   // variational solver used in the smoothing step
    bool has_nan_;

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    vector_t center_;              // mean expansion coefficient vector
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
