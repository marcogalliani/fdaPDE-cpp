//
// Created by Marco Galliani on 04/07/25.
//

#ifndef GAP_FILLER_H
#define GAP_FILLER_H

#include "header_check.h"

namespace fdapde {

// class for handling nan
template <typename VariationalSolver>
    requires(std::is_same_v<typename VariationalSolver::solver_category, ls_solver>)
class GapFiller {
private:
    using smoother_t = std::decay_t<VariationalSolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
    using data_t   = Eigen::Map<const Eigen::Matrix<double, Dynamic, Dynamic, Eigen::ColMajor>>;
public:
    static constexpr int n_lambda = smoother_t::n_lambda;

    template <typename GeoFrame, typename Penalty>
    GapFiller(const std::string& colname, const GeoFrame& gf, Penalty&& penalty) noexcept :
        smoother_(), data_(gf[0].data().template col<double>(colname).as_matrix()){
        fdapde_assert(gf.n_layers() == 1);
        n_locs_ = data_.rows();
        n_units_ = data_.cols();
        if constexpr (requires(Penalty p) { p.get(); }) {
            smoother_ = smoother_t(gf, penalty.get());
        } else {
            smoother_ = smoother_t(gf, penalty(gf.template triangulation<0>()).get());
        }
        n_dofs_ = smoother_.n_dofs();
    }
    auto fit(const std::vector<double>& lambda_grid, int flag) {
        matrix_t X = data_.transpose();         // create temporary of mapped data
        binary_t nan_pattern = na_matrix(X);   // compute missingness pattern
        n_locs_ = X.cols(), n_units_ = X.rows();

        int calibration = (flag & 0b11110);   // detect calibration strategy
        std::vector<double> opt_lambda(n_lambda);
        switch (calibration) {
            case 0: {   // no calibration
                fdapde_assert(lambda_grid.size() == n_lambda);
                std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
            } break;
            case OptimizeGCV: {
                //define the gcv functor
                auto gcv_functor = [&](auto lambda) {
                    const auto& U = solve_(X, nan_pattern, lambda);
                    return gcv_(X, nan_pattern, U, lambda);
                };
                //optimize over the grid
                GridOptimizer<1> optimizer;
                const auto tmp_lambda = optimizer.optimize(gcv_functor,lambda_grid);
                std::copy(tmp_lambda.begin(), tmp_lambda.end(), opt_lambda.begin());
            } break;
            case OptimizeMSRE: {
                std::copy(lambda_grid.begin(), lambda_grid.begin()+n_lambda, opt_lambda.begin());
            } break;
            default: {
                throw std::runtime_error("Unrecognized calibration option.");
            }
        }
        // initialization
        lambda_ = matrix_t::Map(opt_lambda.data(),opt_lambda.size(),1);
        // fit with optimal lambda
        matrix_t U = solve_(X, nan_pattern, opt_lambda);
        return U;
    }
    // observers
    private:
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    const matrix_t solve_(const matrix_t& X, const binary_t& nan_pattern, const LambdaT lambda) {
        using triplet_t = Eigen::Triplet<double>;
        //construct the matrix of weights
        std::vector<int> observed_indexes = nan_pattern.which(false);
        //construct the B matrix: B = ...
        matrix_t design_mat(n_units_,n_units_);
        design_mat = matrix_t::Identity(n_units_,n_units_);
        sparse_matrix_t B_full(n_units_*n_locs_, n_units_*n_dofs_);
        B_full = kronecker(design_mat.sparseView(),smoother_.Psi());
        //remove rows corresponding to unobserved data points
        using triplet_t = Eigen::Triplet<double>;
        std::vector<triplet_t> B_obs_triplets;
        int n_obs = observed_indexes.size();
        int n_cols = B_full.cols();
        for (int i = 0; i < n_obs; ++i) {
            int src_row = observed_indexes[i];
            Eigen::SparseVector<double> row = B_full.row(src_row);
            for (Eigen::SparseVector<double>::InnerIterator it(row); it; ++it) {
                B_obs_triplets.emplace_back(i, it.index(), it.value());
            }
        }
        sparse_matrix_t B_obs(n_obs, n_cols);
        B_obs.setFromTriplets(B_obs_triplets.begin(), B_obs_triplets.end());
        //construct the smoothing matrix
        sparse_matrix_t diag_lambdas(n_units_,n_units_);
        std::vector<triplet_t> lambda_triplets;
        for (int i = 0; i < n_units_; ++i) {
            lambda_triplets.emplace_back(i, i, lambda[0]);
        }
        diag_lambdas.setFromTriplets(lambda_triplets.begin(),lambda_triplets.end());
        //smoothing matrix
        SparseBlockMatrix<double, 2, 2> A(
               B_obs.transpose()*B_obs,                            kronecker(diag_lambdas,smoother_.stiff()),
               kronecker(diag_lambdas,smoother_.stiff()),  kronecker(-diag_lambdas,smoother_.mass())
               );
        using sparse_solver_t = internals::eigen_sparse_solver_movable_wrap<Eigen::SparseLU<sparse_matrix_t>>;
        sparse_solver_t invA;
        //using sparse block matrix
        invA.compute(A);
        if (invA.info() != Eigen::Success) {
            throw std::runtime_error("Matrix factorization failed.");
        }
        //compute the vectorized X
        vector_t vec_X_full = (~nan_pattern).select(X,0).reshaped<Eigen::RowMajor>();
        vector_t vec_X_obs(observed_indexes.size());
        for (int i = 0; i < observed_indexes.size(); ++i) {
            vec_X_obs(i) = vec_X_full(observed_indexes[i]);
        }
        //solve the system: (B^TB + lambda(I_N kron_prod P))y = B^T e (where e is a sampled vector)
        vector_t target = vector_t::Zero(2*B_obs.cols());
        target.head(B_obs.cols()) = B_obs.transpose()*vec_X_obs;
        vector_t y = invA.solve(target);
        vector_t solution = y.head(B_obs.cols());
        //compute the initialisation
        U_.resize(n_units_,n_dofs_);
        U_ = Eigen::Map<Eigen::Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>>(solution.data(), n_units_, n_dofs_);
        return U_;
    }
    // finds vectors s, f minimizing \norm{X - Up}_F^2 + P_{\lambda}(U) and returns the GCV index
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    double gcv_(const matrix_t& X, const binary_t& nan_pattern, const matrix_t& U,const LambdaT lambda) {
        int r = 100; //number of mc simultations
        using triplet_t = Eigen::Triplet<double>;
        //construct the matrix of weights
        std::vector<int> observed_indexes = nan_pattern.which(false);
        std::vector<triplet_t> Dwt_triplets;
        for (int i=0; i <observed_indexes.size(); i++) Dwt_triplets.emplace_back(observed_indexes[i],observed_indexes[i],1.0);
        sparse_matrix_t Dwt(n_units_*n_locs_, n_units_*n_locs_);
        Dwt.setFromTriplets(Dwt_triplets.begin(), Dwt_triplets.end());
        //construct the B matrix: B = ...
        matrix_t design_mat(n_units_,n_units_);
        design_mat = matrix_t::Identity(n_units_,n_units_);
        sparse_matrix_t B(n_units_*n_locs_, n_units_*n_dofs_);
        B = kronecker(design_mat.sparseView(),smoother_.Psi());
        //construct the smoothing matrix
        sparse_matrix_t diag_lambdas(n_units_,n_units_);
        std::vector<triplet_t> lambda_triplets;
        for (int i = 0; i < n_units_; ++i) {
            lambda_triplets.emplace_back(i, i, lambda[0]);
        }
        diag_lambdas.setFromTriplets(lambda_triplets.begin(),lambda_triplets.end());
        //smoothing matrix
        SparseBlockMatrix<double, 2, 2> A(
               B.transpose()*Dwt*B,                            kronecker(diag_lambdas,smoother_.stiff()),
               kronecker(diag_lambdas,smoother_.stiff()),  kronecker(-diag_lambdas,smoother_.mass())
               );
        //Trace estimation
        //sample from the rademacher distribution
        std::mt19937 rng(random_seed);
        rademacher_distribution rademacher;
        matrix_t Us(B.rows(), r);
        for (int i = 0; i < B.rows(); ++i) {
            for (int j = 0; j < r; ++j) { Us(i, j) = rademacher(rng); }
        }
        //solve the system: (B^TB + lambda(I_N kron_prod P))y = B^T e (where e is a sampled vector)
        using sparse_solver_t = internals::eigen_sparse_solver_movable_wrap<Eigen::SparseLU<sparse_matrix_t>>;
        sparse_solver_t invA;
        //using sparse block matrix
        invA.compute(A);
        //building the target
        matrix_t target = matrix_t::Zero(2*B.cols(),r);
        target.topRows(B.cols()) = B.transpose()*Dwt*Us;
        matrix_t y = invA.solve(target);
        //compute e^T*S_m(lambda)*y and estimate the trace by averaging these values
        double trS = 0.0;   // monte carlo Tr[S_m] approximation
        for (int i = 0; i < r; ++i) { trS += Us.col(i).dot(B*y.topRows(B.cols()).col(i)); }
        trS =  trS / r;
        //finally compute the GCV using the estimated dof of the model given by the estimated trace of smoothing matrix
        double dor = n_units_*n_locs_ -  trS;
        double mse = (~nan_pattern).select(X - U*smoother_.Psi().transpose(), 0).squaredNorm();
        double gcv  = n_units_*n_locs_/ std::pow(dor, 2) * mse;
        return gcv;
    }

public:
    const matrix_t& U() const { return U_; }   // reconstructed matrix
    const matrix_t& lambda() const { return lambda_; }

    data_t data_;           // mapped geoframe data
    matrix_t U_;

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    int n_folds_ = 5;
    smoother_t smoother_;
    matrix_t lambda_;              // selected PCs smoothing level
    };

// deduction guide
template <typename GeoFrame, typename Penalty>
GapFiller(const std::string& colname, const GeoFrame& gf, Penalty&& solver) -> GapFiller<typename Penalty::solver_t>;

}


#endif //GAP_FILLER_H
