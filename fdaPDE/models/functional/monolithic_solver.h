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

template<typename ModelType, typename SVDType_=RSVD<DMatrix<double>>>
class FixedLambdaMonolithicSolver{
private:
    ModelType &model_;
    int rank_;
    DVector<double> lambda_;
    int seed_ = fdapde::random_seed;
    //Solutions to the fPCA problem
    DMatrix<double> invD_; //factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
    DMatrix<double> scores_;
    DMatrix<double> loadings_;
public:
    FixedLambdaMonolithicSolver(ModelType &model, int seed) : model_(model), seed_(seed){}

    void init(const DVector<double> &lambda){
        if(lambda_.size()!=0 && lambda_==lambda) return;
        lambda_ = lambda;
        Eigen::LLT<DMatrix<double>> chol(model_.Psi().transpose()*model_.Psi() + model_.P(lambda_));
        invD_ = chol.matrixL().solve(DMatrix<double>::Identity(model_.Psi().cols(),model_.Psi().cols()));
        return;
    }
    void compute(const DMatrix<double> &X, int rank){
        rank_ = rank;
        // compute SVD of X*\Psi*(D^{-1})^\top
        SVDType_ svd; //here we may use a randomized algorithm
        if constexpr (is_rand_svd<SVDType_>{}){
            svd.set_seed(seed_);
            svd.compute(X*model_.Psi()*invD_.transpose(),rank_);
        } else{
            svd.compute(X*model_.Psi()*invD_.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        scores_ = svd.matrixU().leftCols(rank_)*svd.singularValues().head(rank_).asDiagonal();
        loadings_ = (svd.matrixV().leftCols(rank_).transpose()*invD_).transpose();
        return;
    }
    //gcv score (has to be preceded by a call to compute())
    double gcv(double edf_discount=1.0){
        DMatrix<double> S_m = model_.Psi()*loadings_;
        S_m = S_m*S_m.transpose();
        double gcv_score = model_.X().cols()/std::pow(model_.X().cols()-edf_discount*S_m.trace(),2)*
                           (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm();
        return gcv_score;
    }
    //test the reconstruction on a different portion of the data (has to be preceded by a call to compute())
    double reconstruction_error(const DMatrix<double>& X_test){
        DMatrix<double> S_m = loadings_;
        S_m = model_.Psi()*S_m;
        S_m = S_m*S_m.transpose();

        return (X_test*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm() / (X_test.rows() * X_test.cols());
    }
    //setters
    void setLambda(DVector<double> lambda){
        init(lambda); //the solver has to be reinitialised whenever a new lambda is set
        return;
    }
    //getters
    const DMatrix<double>& scores() const { return scores_; }
    const DMatrix<double>& loadings() const { return loadings_; }
};

#endif //MONOLITHIC_SOLVER_H
