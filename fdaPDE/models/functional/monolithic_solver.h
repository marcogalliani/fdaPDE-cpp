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

#ifndef MONOLITHIC_SOLVER_H
#define MONOLITHIC_SOLVER_H

#include "fpca.h"
#include <Eigen/Eigenvalues>

template<typename ModelType, typename SVDType=RSVD<DMatrix<double>>>
class MonolithicSolver{
private:
    const ModelType &model_;
    int rank_;
    DVector<double> lambda_;
    int seed_ = fdapde::random_seed;
    //Solutions to the generalized eigenvalue problem
    Eigen::SelfAdjointEigenSolver<DMatrix<double>> evd_;
    Eigen::SimplicialLLT<SpMatrix<double>> chol_; //cholesky of Psi^T*Psi
    DMatrix<double> invL_; //inverse of the cholesky factor
    DMatrix<double> V_; //generalized eigenvectors
    //Solutions to the fPCA problem
    SVDType svd_; //here we may use a randomized algorithm
    DMatrix<double> invD_; //factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
public:
    MonolithicSolver() = delete; //model has to be initialised
    MonolithicSolver(const ModelType &model, int seed) : model_(model), seed_(seed){}

    void init(){
        //Genelarized eigenvalue problem: P*V = (Psi^T*Psi)*V*Lambda
        chol_.compute(model_.Psi().transpose()*model_.Psi());
        invL_ = chol_.matrixL().solve(DMatrix<double>::Identity(model_.n_basis(),model_.n_basis()))*chol_.permutationP().toDenseMatrix().template cast<double>();
        evd_.compute(invL_*model_.P(DVector<double>::Ones(1))*invL_.transpose());
        V_ = invL_.transpose()*evd_.eigenvectors();
        return;
    }
    void compute(const DMatrix<double> &X, int rank, DVector<double> lambda){
        rank_ = rank;
        if (lambda_!=lambda){
            lambda_ = lambda;
            // assemble the factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
            invD_ = (DVector<double>::Ones(model_.n_basis())+lambda_(0)*evd_.eigenvalues()).unaryExpr([](double x){ return 1/std::sqrt(x);}).asDiagonal()*V_.transpose();
        };
        // compute SVD of X*\Psi*(D^{-1})^\top
        const auto start{std::chrono::steady_clock::now()};
        if constexpr (is_rand_svd<SVDType>{}){
            svd_.compute(X*model_.Psi()*invD_.transpose(),rank_);
        } else{
            svd_.compute(X*model_.Psi()*invD_.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        const auto end{std::chrono::steady_clock::now()};
        std::ofstream svd_time("results/svd_time.csv");
        svd_time  << (std::chrono::duration<double>{end - start}).count() << std::endl;
        svd_time.close();
        return;
    }
    //gcv score (has to be preceded by a call to compute())
    double gcv(double edf_discount=1.0){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();
        double gcv_score = model_.X().cols()/std::pow(model_.X().cols()-edf_discount*S_m.trace(),2)*
                           (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm();
        return gcv_score;
    }
    //test the reconstruction on a different portion of the data (has to be preceded by a call to compute())
    double reconstruction_error(const DMatrix<double>& X_test){
        DMatrix<double> S_m = invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = model_.Psi()*S_m;
        S_m = S_m*S_m.transpose();

        return (X_test*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm() / (X_test.rows() * X_test.cols());
    }
    //getters
    const DMatrix<double> scores() const { return svd_.matrixU().leftCols(rank_); }
    const DMatrix<double> loadings() const { return (svd_.singularValues().head(rank_).asDiagonal()*svd_.matrixV().leftCols(rank_).transpose()*invD_).transpose(); }
};


template<typename ModelType, typename SVDType=RSVD<DMatrix<double>>>
class FixedLambdaMonolithicSolver{
private:
    const ModelType &model_;
    int rank_;
    DVector<double> lambda_;
    int seed_ = fdapde::random_seed;
    //Solutions to the fPCA problem
    SVDType svd_; //here we may use a randomized algorithm
    DMatrix<double> invD_; //factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
public:
    FixedLambdaMonolithicSolver() = delete; //model has to be initialised
    FixedLambdaMonolithicSolver(const ModelType &model, int seed) : model_(model), seed_(seed){}

    void init(DVector<double> lambda){
        lambda_ = lambda;
        Eigen::LLT<DMatrix<double>> chol(model_.Psi().transpose()*model_.Psi() + model_.P(lambda));
        invD_ = chol.matrixL().solve(DMatrix<double>::Identity(model_.n_basis(),model_.n_basis()));
        return;
    }
    void compute(const DMatrix<double> &X, int rank, DVector<double> lambda){
        if(lambda_ != lambda) init(lambda);
        rank_ = rank;
        // compute SVD of X*\Psi*(D^{-1})^\top
        const auto start{std::chrono::steady_clock::now()};
        if constexpr (is_rand_svd<SVDType>{}){
            svd_.compute(X*model_.Psi()*invD_.transpose(),rank_);
        } else{
            svd_.compute(X*model_.Psi()*invD_.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        const auto end{std::chrono::steady_clock::now()};
        std::ofstream svd_time("results/svd_time.csv");
        svd_time  << (std::chrono::duration<double>{end - start}).count() << std::endl;
        svd_time.close();
        return;
    }
    //gcv score (has to be preceded by a call to compute())
    double gcv(double edf_discount=1.0){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();
        double gcv_score = model_.X().cols()/std::pow(model_.X().cols()-edf_discount*S_m.trace(),2)*
                           (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm();
        return gcv_score;
    }
    //test the reconstruction on a different portion of the data (has to be preceded by a call to compute())
    double reconstruction_error(const DMatrix<double>& X_test){
        DMatrix<double> S_m = invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = model_.Psi()*S_m;
        S_m = S_m*S_m.transpose();

        return (X_test*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm() / (X_test.rows() * X_test.cols());
    }
    //getters
    const DMatrix<double> scores() const { return svd_.matrixU().leftCols(rank_); }
    const DMatrix<double> loadings() const { return (svd_.singularValues().head(rank_).asDiagonal()*svd_.matrixV().leftCols(rank_).transpose()*invD_).transpose(); }
};

#endif //MONOLITHIC_SOLVER_H
