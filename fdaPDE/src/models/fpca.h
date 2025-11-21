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

// bit 3 reserved for mean computation
[[maybe_unused]] constexpr int DoNotComputeMean = 0x1 << 3; // bit 3

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
            Eigen::Matrix<double, n_lambda, 1> opt_lambda;
            switch (calibration) {
            case 0: {   // no calibration
                fdapde_assert(lambda_grid.size() == n_lambda);
                std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
            } break;
            case OptimizeGCV: {
                auto gcv_functor = [&](auto lambda) { return gcv_(X, lambda, V.col(i)); };
                GridSearch<n_lambda> optimizer;
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
        std::array<double, n_lambda> lambda_vec;
        std::copy(lambda.data(), lambda.data() + n_lambda, lambda_vec.begin());
        if (edf_map_.find(lambda_vec) == edf_map_.end()) {   // cache Tr[S]
            edf_map_[lambda_vec] = smoother_->edf();
        }
        int dor = n_locs_ - edf_map_.at(lambda_vec);
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
        Eigen::Matrix<double, n_lambda, 1> opt_lambda;
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
        } break;
        case OptimizeGCV: {
            auto gcv_functor = [&](auto lambda) { return gcv_(X, rank, lambda, V); };
            GridSearch<n_lambda> optimizer;
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
    std::unordered_map<std::array<double, n_lambda>, double, internals::std_array_hash<double, n_lambda>> edf_map_;
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
        Eigen::Matrix<double, n_lambda, 1> opt_lambda;
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
        } break;
        case OptimizeGCV: {
            auto gcv_functor = [&](auto lambda) { return gcv_(X, rank, lambda, flag); };
            GridSearch<n_lambda> optimizer;
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
        int dor = n_locs_ -  (invD_*smoother_->Psi().transpose()).squaredNorm();
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


// utilities to handle k-fold cross-validation in the case of missing data
class k_fold_missing_cv_impl {
private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    using fold_t = std::vector<std::pair<int, int>>;
    using TrainTestPartition = std::pair<binary_t, binary_t>;

    int n_folds_ = 5;
    std::vector<fold_t> folds_;

    // utility to generate non-overlapping folds
    void generate_folds(const matrix_t& X, int K){
        int n = X.rows(), m = X.cols();
        std::vector<std::pair<int, int>> duplets;
        // collect all observed (i, j) pairs
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                if (!std::isnan(X(i, j))) duplets.emplace_back(i, j);
        // shuffle once
        std::shuffle(duplets.begin(), duplets.end(), std::default_random_engine{});
        // partition into K folds
        folds_.resize(K);
        for (std::size_t i = 0; i < duplets.size(); ++i)
            folds_[i % K].push_back(duplets[i]);
    }
public:
    k_fold_missing_cv_impl(const matrix_t &X, int K) : n_folds_(K){
        generate_folds(X, K);
    }
    //utility to access the i-th fold
    std::pair<binary_t, binary_t> split(const matrix_t& X, int fold_index){
        int n = X.rows(), m = X.cols();
        binary_t test_mask(n, m), train_mask(n, m);
        for (int k = 0; k < folds_.size(); ++k) {
            for (const auto& [i, j] : folds_[k]) {
                if (k == fold_index)
                    test_mask.set(i, j);
                else
                    train_mask.set(i, j);
            }
        }
        return {train_mask, test_mask};
    }
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

        int calibration = (flag & 0b00110);   // detect calibration strategy
        std::vector<double> opt_lambda_mu(n_lambda);
        std::vector<double> opt_lambda_F(n_lambda);

        matrix_t Un = matrix_t::Zero(n_units_, n_locs_);
        Un.rowwise() += (~nan_pattern).select(X,0).colwise().mean(); // start by imputing the mean
        int opt_rank;
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda_mu.begin());
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda_F.begin());
            opt_rank = rank;
        } break;
        case OptimizeGCV: {
            //define the gcv functor
            auto gcv_functor = [&](auto params) {
                std::vector<double> lambda_mu(n_lambda);
                std::vector<double> lambda_F(n_lambda);
                std::copy(params.begin(), params.begin()+n_lambda, lambda_mu.begin());
                std::copy(params.begin()+n_lambda, params.begin()+2*n_lambda, lambda_F.begin());

                Un = matrix_t::Zero(n_units_, n_locs_);
                Un.rowwise() += (~nan_pattern).select(X,0).colwise().mean(); // start by imputing the mean;
                matrix_t center, F, S;
                for (int k = 1; k <= rank; ++k) {
                    const auto& [c, f, s] = solve_(X, nan_pattern, Un, k, lambda_mu,lambda_F, flag & 0b1001);
                    Un = s * f.transpose() * smoother_.Psi().transpose(); // reconstruction update
                    Un.rowwise() += c.transpose() * smoother_.Psi().transpose();
                    center = c; S=s; F=f;
                }
                double edf = edf_(nan_pattern, center.col(0), S, F, rank, lambda_mu,lambda_F);
                double dor = n_units_*n_locs_ - edf;
                double gcv  = n_units_*n_locs_/ std::pow(dor, 2) * (~nan_pattern).select(X - Un, 0).squaredNorm();;

                return gcv;
            };
            Eigen::Matrix<double, Dynamic, Dynamic, Eigen::RowMajor> grid_2D(lambda_grid.size(), 2);
            int idx = 0;
            for (int i = 0; i < lambda_grid.size(); ++i) {
                //for (int j = 0; j < lambda_grid.size(); ++j) {
                    //for(int k = 1; k <= rank; k++){
                        grid_2D(idx, 0) = lambda_grid[i];
                        grid_2D(idx, 1) = lambda_grid[i];
                        ++idx;
                    //}
                //}
            }
            //optimize over the grid
            GridSearch<2> optimizer;
            auto optimal_lambdas = optimizer.optimize(gcv_functor,grid_2D);
            // resize to have shape: grid_sz-by-n_lambda
            gcv_scores_.resize(lambda_grid.size(),1);
            for (int i = 0; i < lambda_grid.size(); ++i) {
                //for (int j = 0; j < lambda_grid.size(); j++){
                    gcv_scores_(i, 0) = optimizer.values()[i];
                //}
            }
            std::copy(optimal_lambdas.begin(), optimal_lambdas.begin()+n_lambda, opt_lambda_mu.begin());
            std::copy(optimal_lambdas.begin()+n_lambda, optimal_lambdas.begin()+2*n_lambda, opt_lambda_F.begin());
            opt_rank = rank; //optimal_lambdas[optimal_lambdas.size()-1];
        } break;
        case OptimizeMSRE: {
            auto k_fold_cv = k_fold_missing_cv_impl(X,n_folds_);
            // store the rmse estimated via KCV for each pair (lambda,rank)
            matrix_t rmse_table = matrix_t::Zero(lambda_grid.size(),rank);
            for(int i = 0; i < lambda_grid.size(); ++i) {
                matrix_t mse_per_fold = matrix_t::Zero(n_folds_, rank);
                for(int j = 0; j < n_folds_; ++j) {
                    auto [train_mask, test_mask] = k_fold_cv.split(X, j);
                    matrix_t X_train = train_mask.select(X,std::numeric_limits<double>::quiet_NaN());
                    binary_t train_nan_pattern = na_matrix(X_train);
                    // repeat for increasing rank
                    Un = matrix_t::Zero(n_units_, n_locs_);
                    Un.rowwise() += (~nan_pattern).select(X,0).colwise().mean(); // start by imputing the mean;
                    for (int k = 1; k <= rank; ++k) {
                        // TEMPORARY FIX: std::vector<double>{double} is not nice
                        const auto& [mu, F, S] = solve_(X_train, train_nan_pattern, Un,k, std::vector<double>{lambda_grid[i]},std::vector<double>{lambda_grid[i]}, flag & flag & 0b1001);
                        Un = S * F.transpose()*smoother_.Psi().transpose();
                        Un.rowwise() += mu.transpose()*smoother_.Psi().transpose(); // reconstruction update
                        // check the loss for each k in {1,...,rank}
                        double loss = test_mask.select(X-Un,0).squaredNorm();
                        double n_test_obs = test_mask.count();
                        //average for the number of observations
                        mse_per_fold(j,k-1) = loss/n_test_obs;
                    } // rank-recursion
                } //iteration over the folds
                rmse_table.row(i) = mse_per_fold.colwise().mean();
            } // iteration over the lambda
            rmse_table = rmse_table.array() / n_folds_;
            Eigen::Index opt_lambda_idx, opt_rank_idx;
            rmse_table.minCoeff(&opt_lambda_idx, &opt_rank_idx);
            opt_lambda_mu = opt_lambda_F = std::vector<double>{lambda_grid[opt_lambda_idx]};
            opt_rank = opt_rank_idx + 1; //rank grid is always {1,...,rank}
            // gcv scores (not really gcv, just kcv)
            gcv_scores_.resize(lambda_grid.size(),1);
            gcv_scores_.col(0) = rmse_table.col(opt_rank_idx);
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
        lambda_.resize(opt_rank+1, n_lambda);
        // fit with optimal lambda
        Un = matrix_t::Zero(n_units_, n_locs_);
        Un.rowwise() += (~nan_pattern).select(X,0).colwise().mean(); // start by imputing the mean;
        for (int k = 1; k <= opt_rank; ++k) {
            const auto& [center, F, S] = solve_(X, nan_pattern, Un,k, opt_lambda_mu,opt_lambda_F, flag & flag & 0b1001);
            Un = S * F.transpose()*smoother_.Psi().transpose(); // reconstruction update
            Un.rowwise() += center.transpose()*smoother_.Psi().transpose();
            center_ = center;
            f_.leftCols(k) = F;
            s_.leftCols(k) = S;
        }
        // store results
        for (int j = 0; j < n_lambda; ++j) { lambda_(0, j) = opt_lambda_mu[j]; }
        for (int i = 0; i < opt_rank; ++i) {
            for (int j = 0; j < n_lambda; ++j) { lambda_(i+1, j) = opt_lambda_F[j]; }
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
    const matrix_t& gcv_scores() const { return gcv_scores_; }
   private:

    // the solve_ method implements the MM scheme (the rank recursion is performed during the fit)
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, const binary_t& nan, const matrix_t& Un0, int rank, const LambdaT lambda_mu, const LambdaT lambda_F, int flag) {
        for (int i = 0; i < lambda_mu.size(); ++i) { fdapde_assert(lambda_mu[i] > 0); fdapde_assert(lambda_F[i] > 0);}
        vector_t center(n_dofs_);
        matrix_t F(n_dofs_ , rank);
	    matrix_t S(n_units_, rank);
        matrix_t U;
	    matrix_t Un = Un0;
        // define the mean function used to center the data in the mean estimation step
        std::function<vector_t(const matrix_t &, double)> mean_functor;
        bool computeMean = !(flag & DoNotComputeMean);
        if (computeMean) {
            mean_functor = [&](const matrix_t& X, double lambda) {
                smoother_.update_response(X.colwise().mean().transpose());
                smoother_.fit(lambda_mu[0]);
                return smoother_.f();
            };
        } else {
            mean_functor = [&](const matrix_t& X, double lambda) {
                return vector_t::Zero(n_dofs_);
            };
        }
        // iterate until the MM scheme converges
        int n_iter = 0;
        double Jold = std::numeric_limits<double>::max(), Jnew = 1.0;
        while (!almost_equal(Jnew, Jold, tol_) && n_iter < max_iter_) {
            // imputation update
            matrix_t Xn = (~nan).select(X, Un);
            const auto mu = mean_functor(Xn, lambda_mu[0]);
            Xn.rowwise() -= (smoother_.Psi() * mu).transpose();
            // rank-k fPCA on imputed data
            auto [f, s] = fpca_->fit(Xn.transpose(), rank, lambda_F, flag & 0x1); //always run with no calibration
            // return orthonormal scores
            for (int i = 0; i < rank; ++i) {
                s.col(i) = s.col(i) / fpca_->loadings_norm()[i];
                f.col(i) = f.col(i) * fpca_->loadings_norm()[i];
            }
            U = s * f.transpose();   // reconstruction update
            U.rowwise() += mu.transpose();
            // prepare for next iteration
            n_iter++;
            Jold = Jnew;
            Un = U * smoother_.Psi().transpose();
            Jnew = ((~nan).select(X - Un, 0)).squaredNorm() + n_units_*mu.transpose()*smoother_.P(lambda_mu)*mu + (f.transpose() * smoother_.P(lambda_F) * f).trace();
            //double edf = edf_(nan, mu, S, F, rank, lambda_F);
            //Jnew = n_units_*n_locs_*((~nan).select(X - Un, 0)).squaredNorm()/std::pow(n_units_*n_locs_ - edf,2);
            if (almost_equal(Jnew, Jold, tol_) || n_iter == max_iter_) {
                center = mu; F = f; S = s;
            }
        }
        if (n_iter>=max_iter_){ std::cout << "Convergence issues, relative error: " <<  (std::fabs(Jnew) < std::fabs(Jold) ? std::fabs(Jnew-Jold)/Jold : std::fabs(Jnew-Jold)/Jnew) << std::endl;}
	    return std::make_tuple(center, F, S);
    }
    // compute the effective degrees of freedom of the method (assume a unique lambda)
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double edf_(const binary_t& nan_pattern, const vector_t& center, const matrix_t& S, const matrix_t& F, int rank, const LambdaT lambda_mu, const LambdaT lambda_F) {
        matrix_t U = S * F.transpose(); // reconstruction
        U.rowwise() += center.transpose();
        using triplet_t = Eigen::Triplet<double>;
        //construct the matrix of weights
        std::vector<int> observed_indexes = nan_pattern.which(false); // rowmajor ordering
        std::vector<triplet_t> Dwt_triplets;
        /*
        for (int i=0; i <observed_indexes.size(); i++) Dwt_triplets.emplace_back(observed_indexes[i],observed_indexes[i],1.0);
        sparse_matrix_t Dwt(n_units_*n_locs_, n_units_*n_locs_);
        Dwt.setFromTriplets(Dwt_triplets.begin(), Dwt_triplets.end());
        */
        //construct the B matrix: B = [1/\sqrt{N} ones(N) kron_prod \Psi, S kron_prod Psi]
        //matrix_t design_mat(n_units_,n_units_+1);
        matrix_t design_mat(n_units_,rank+1);
        design_mat.col(0) = 1/std::sqrt(n_units_)*vector_t::Ones(n_units_);
        //design_mat.rightCols(n_units_) = matrix_t::Identity(n_units_,n_units_);
        design_mat.rightCols(rank) = S;
        //sparse_matrix_t B(n_units_*n_locs_, (n_units_+1)*n_dofs_);
        sparse_matrix_t B(n_units_*n_locs_, (rank+1)*n_dofs_);
        B = kronecker(design_mat.sparseView(),smoother_.Psi());
        // ALTERNATIVE: construct the B_obs matrix
        // You already have observed_indexes (N_obs = observed_indexes.size())
        // B is your original, full sparse matrix
        // 1. Create the fast lookup map (old_row -> new_row)
        std::unordered_map<int, int> row_map;
        row_map.reserve(observed_indexes.size());
        for (int i = 0; i < observed_indexes.size(); ++i) {
            row_map[observed_indexes[i]] = i; // Map old_index -> new_index
        }
        // 2. Build B_obs by filtering B's non-zeros
        std::vector<triplet_t> B_obs_triplets;
        B_obs_triplets.reserve(B.nonZeros()); // Over-estimate, but fast
        // Iterate over B (assuming B is ColMajor, which is Eigen's default)
        for (int k = 0; k < B.outerSize(); ++k) { // Iterates over columns
            for (typename sparse_matrix_t::InnerIterator it(B, k); it; ++it) {
                auto map_it = row_map.find(it.row()); // Check if this row is observed
                if (map_it != row_map.end()) {
                    // It is! Add it to our new matrix's triplets
                    // map_it->second is the new, compressed row index
                    B_obs_triplets.emplace_back(map_it->second, it.col(), it.value());
                }
            }
        }
        // 3. Create the new, smaller matrix
        sparse_matrix_t B_obs(observed_indexes.size(), B.cols());
        B_obs.setFromTriplets(B_obs_triplets.begin(), B_obs_triplets.end());
        //construct the smoothing matrix
        //sparse_matrix_t diag_lambdas(n_units_+1,n_units_+1);
        sparse_matrix_t diag_lambdas(rank+1,rank+1);
        std::vector<triplet_t> lambda_triplets;
        lambda_triplets.emplace_back(0, 0, lambda_mu[0]);
        //for (int i = 0; i < n_units_; ++i) {
        for (int i = 0; i < rank; ++i) {
            lambda_triplets.emplace_back(i+1, i+1, lambda_F[0]);
        }
        diag_lambdas.setFromTriplets(lambda_triplets.begin(),lambda_triplets.end());
        //smoothing matrix
        SparseBlockMatrix<double, 2, 2> A(
               B_obs.transpose()*B_obs /*B.transpose()*Dwt*B*/, kronecker(diag_lambdas,smoother_.stiff()),
               kronecker(diag_lambdas,smoother_.stiff()), kronecker(-diag_lambdas,smoother_.mass())
               );
        //Trace estimation
        //sample from the rademacher distribution
        std::mt19937 rng(random_seed);
        rademacher_distribution rademacher;
        matrix_t Us(B.rows(), n_mc_samples_);
        for (int i = 0; i < B.rows(); ++i) {
            for (int j = 0; j < n_mc_samples_; ++j) { Us(i, j) = rademacher(rng); }
        }
        //solve the system: (B^TB + lambda(I_N kron_prod P))y = B^T e (where e is a sampled vector)
        using sparse_solver_t = eigen_sparse_solver_movable_wrap<Eigen::SparseLU<sparse_matrix_t>>;
        sparse_solver_t invA;
        //using sparse block matrix
        invA.compute(A);
        //building the target
        matrix_t target = matrix_t::Zero(2*B.cols(),n_mc_samples_);
        target.topRows(B.cols()) = B.transpose()*Us;//B.transpose()*Dwt*Us;
        matrix_t y = invA.solve(target);
        //compute e^T*S_m(lambda)*y and estimate the trace by averaging these values
        double trS = 0.0;   // monte carlo Tr[S_m] approximation
        for (int i = 0; i < n_mc_samples_; ++i) { trS += Us.col(i).dot(B*y.topRows(B.cols()).col(i)); }//Us.col(i).dot(Dwt*B*y.topRows(B.cols()).col(i)); }
        trS =  trS / n_mc_samples_;
        return trS;
    }
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    int n_folds_ = 5;
    int n_mc_samples_ = 100;       // to estimate the trace of the smoothing matrix
    fpca_t* fpca_;
    smoother_t smoother_;
    vector_t center_;              // mean expansion coefficient vector
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated PCs
    matrix_t lambda_;              // selected PCs smoothing level
    matrix_t gcv_scores_;          // gcv scores (#lambda_grid-by-rank matrix)

    // MM scheme parameters
    double tol_ = 1e-4;
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
            // default to OptimMSRE calibration, if no calibration provided
            if (lambda_grid.size() > n_lambda && (flag & 0b11110) == 0) { flag = flag | OptimizeMSRE; }
            internals::fpca_na_impl mm_scheme(solver_);
            const auto& [mu,f, s] = mm_scheme.fit(data_, rank, lambda_grid, flag);
            center_ = std::move(mu);
            f_ = std::move(f);
            s_ = std::move(s);
            f_norm_ = solver_.loadings_norm();
            lambda_ = mm_scheme.lambda();
            gcv_scores_ = mm_scheme.gcv_scores();
        } else {
            // default to OptimGCV calibration, if no calibration provided
            if (lambda_grid.size() > n_lambda && (flag & 0b11110) == 0) { flag = flag | OptimizeGCV; }
            // data centering
            matrix_t centred_data = data_.transpose();
            bool computeMean = !(flag & DoNotComputeMean);
            if (computeMean) {
                smoother_.update_response(centred_data.colwise().mean());
                GridSearch<1> opt;
                auto gcv_functor = [&](auto lambda) {
                    smoother_.fit(lambda);
                    double dor =  n_locs_ - smoother_.edf();  // residual degrees of freedom
                    return (n_locs_ / std::pow(dor, 2)) * (~smoother_.nan_pattern()).select(smoother_.fn() - smoother_.response(),0).squaredNorm();
                };
                opt.optimize(gcv_functor, lambda_grid);
                smoother_.fit(opt.optimum());
                center_ = std::move(smoother_.f());
                centred_data.rowwise() -= smoother_.fn().transpose();
            }
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
    const vector_t& center_locs() const { return smoother_.Psi() * center_;} // mean vector
    const matrix_t& S() const { return s_; }   // scoring matrix
    const matrix_t& F() const { return f_; }   // loading matrix
    matrix_t Fn() const { return smoother_.Psi() * f_; }
    const std::vector<double>& loadings_norm() const { return f_norm_; }
    const matrix_t& lambda() const { return lambda_; }
    const matrix_t& gcv_scores() const { return gcv_scores_; }

   private:
    matrix_t data_;         // mapped geoframe data
    smoother_t smoother_;   // variational solver used in the smoothing step
    bool has_nan_;

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    vector_t center_;              // mean expansion coefficient vector
    matrix_t f_;                   // PCs expansion coefficient vector
    matrix_t s_;                   // PCs scores
    std::vector<double> f_norm_;   // L^2 norm of estimated components
    matrix_t lambda_;              // selected level of smoothing for each component
    matrix_t gcv_scores_;          // gcv scores (#lambda_grid-by-rank matrix)
};

// deduction guide
template <typename GeoFrame, typename Penalty>
fPCA(const std::string& colname, const GeoFrame& gf, Penalty&& solver) -> fPCA<typename Penalty::solver_t>;

}   // namespace fdapde

#endif   // __FPCA_H__
