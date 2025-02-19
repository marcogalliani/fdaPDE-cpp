//
// Created by Marco Galliani on 03/12/24.
//
#include "fdaPDE/models/functional/fpca.h"
using fdapde::models::FPCA;
using fdapde::models::RegularizedSVD;
using fdapde::models::Sampling;

//Eigen
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

//new implementation: efficient evaluation over a grid of lambda
template<typename ModelType, typename SVDType=RSVD<DMatrix<double>>>
class MonoFPCASolver{
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
    DMatrix<double> D_;
    DMatrix<double> invD_; //factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
    //Time report
    double factorization_time_;
    double svd_time_;

public:
    MonoFPCASolver(const ModelType &model) : model_(model){
        const auto start{std::chrono::steady_clock::now()};
        //Genelarized eigenvalue problem: P*V = (Psi^T*Psi)*V*Lambda
        chol_.compute(model.Psi().transpose()*model.Psi());
        invL_ = chol_.matrixL().solve(DMatrix<double>::Identity(model.n_basis(),model.n_basis()))*chol_.permutationP();
        evd_.compute(invL_*model.P(DVector<double>::Ones(1))*invL_.transpose());
        V_ = invL_.transpose()*evd_.eigenvectors();
        const auto end{std::chrono::steady_clock::now()};
        factorization_time_ = (std::chrono::duration<double>{end - start}).count();
    }
    void compute(const DMatrix<double> &X, int rank, double lambda){
        const auto start{std::chrono::steady_clock::now()};
        rank_ = rank;
        // assemble the factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
        D_ = (chol_.permutationPinv()*chol_.matrixL().toDense())*evd_.eigenvectors()*(DVector<double>::Ones(model_.n_basis())+lambda*evd_.eigenvalues()).unaryExpr([](double x){ return std::sqrt(x);}).asDiagonal();
        invD_ = (DVector<double>::Ones(model_.n_basis())+lambda*evd_.eigenvalues()).unaryExpr([](double x){ return 1/std::sqrt(x);}).asDiagonal()*V_.transpose();
        // compute SVD of X*\Psi*(D^{-1})^\top
        if constexpr (is_rand_svd<SVDType>{}){
            svd_.compute(X*model_.Psi()*invD_.transpose(),rank_);
        } else{
            svd_.compute(X*model_.Psi()*invD_.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        const auto end{std::chrono::steady_clock::now()};
        svd_time_ = (std::chrono::duration<double>{end - start}).count();
        return;
    }
    //gcv score (has to be preceded by a call to compute())
    double gcv(double edf_discount=1.0){
        DMatrix<double> loadings = invD_.transpose()*svd_.matrixV().leftCols(rank_);
        DMatrix<double> S_m = model_.Psi()*loadings*loadings.transpose()*model_.Psi().transpose();

        double gcv_score = model_.X().cols()/std::pow(model_.X().cols()-edf_discount*S_m.trace(),2)*
                           (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm();
        return gcv_score;
    }
    double full_smoothing_gcv(double edf_discount=1.0){
        DMatrix<double> loadings = invD_.transpose()*svd_.matrixV().leftCols(rank_);

        DMatrix<double> smoothed_data = svd_.matrixU().leftCols(rank_).transpose()*model_.X();
        DMatrix<double> S_m = model_.Psi()*invD_.transpose();
        S_m = S_m*S_m.transpose();

        int n_locs = model_.X().cols();
        double gcv_score = n_locs/std::pow(n_locs-edf_discount*S_m.trace(),2)*
                           (smoothed_data*(DMatrix<double>::Identity(n_locs, n_locs) - S_m)).squaredNorm();
        return gcv_score;
    }
    //ocv score (has to be preceded by a call to compute())
    double ocv(){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();

        DVector<double> pred_err = (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).colwise().squaredNorm();
        double ocv_score = (
                pred_err.array()
                / (1 - S_m.diagonal().array()).square()
        ).sum();
        return ocv_score;
    }
    //test the reconstruction on a different portion of the data (has to be preceded by a call to compute())
    double reconstruction_error(const DMatrix<double> &X_test){

        DMatrix<double> loadings = invD_.transpose()*svd_.matrixV().leftCols(rank_);
        DVector<double> loadings_norm = (svd_.matrixV().leftCols(rank_)).colwise().norm();
        loadings = loadings.array().rowwise() / loadings_norm.transpose().array();

        DMatrix<double> S_m = model_.Psi()*(invL_.transpose()*invL_)*(D_*D_.transpose());
        S_m = S_m*loadings*loadings.transpose();
        S_m = S_m*model_.Psi().transpose();

        //DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        //S_m = S_m*S_m.transpose();

        return (X_test*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)
        ).squaredNorm() / (X_test.rows() * X_test.cols());
    }
    double reconstruction_error_projection(const DMatrix<double> &X_test){
        DMatrix<double> loadings = invD_.transpose()*svd_.matrixV().leftCols(rank_);
        DMatrix<double> scores = X_test*model_.Psi()*loadings*(loadings.transpose()*(model_.Psi().transpose()*model_.Psi())*loadings).inverse();

        return (X_test-scores*loadings.transpose()*model_.Psi().transpose()).squaredNorm()/(X_test.rows() * X_test.cols());
    }
    double reconstruction_error_simple(const DMatrix<double> &X_test){
        DMatrix<double> loadings = invD_.transpose()*svd_.matrixV().leftCols(rank_);
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();

        return (X_test*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)
               ).norm() / std::sqrt(X_test.rows() * X_test.cols());
    }
    //getters
    const DMatrix<double> scores() const { return svd_.matrixU().leftCols(rank_); }
    const DMatrix<double> loadings() const { return (svd_.singularValues().head(rank_).asDiagonal()*svd_.matrixV().leftCols(rank_).transpose()*invD_).transpose(); }
    //time report
    double factorization_time() const{ return factorization_time_;}
    double fit_time() const{ return svd_time_;}
};

//old implemantation: efficient for fixed lambda
template<typename ModelType, typename SVDType=RSVD<DMatrix<double>>>
class FixedSmoothingMonoFPCASolver{
private:
    const ModelType &model_;
    int rank_;
    double lambda_;
    //Factorization of the smoothing matrix
    Eigen::LLT<DMatrix<double>> chol_; //cholesky of Psi^T*Psi
    DMatrix<double> invD_; //factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
    //Solutions to the fPCA problem
    SVDType svd_; //here we use a randomized algorithm
    //Time report
    double factorization_time_;
    double svd_time_;

public:
    FixedSmoothingMonoFPCASolver(const ModelType &model, double lambda) : model_(model), lambda_(lambda){
        const auto start{std::chrono::steady_clock::now()};
        //factorization of the smoothing matrix
        chol_.compute(model.Psi().transpose()*model.Psi()+model.P(DVector<double>::Constant(1,lambda)));
        invD_ = chol_.matrixL().solve(DMatrix<double>::Identity(model.n_basis(),model.n_basis()));
        const auto end{std::chrono::steady_clock::now()};

        factorization_time_ = (std::chrono::duration<double>{end - start}).count();
    }
    void compute(const fdapde::core::BinaryMatrix<Dynamic,Dynamic>& train_set, int rank, double lambda){
        const auto start{std::chrono::steady_clock::now()};
        rank_ = rank;
        if(lambda_ != lambda){
            lambda_ = lambda;
            chol_.compute(model_.Psi().transpose()*model_.Psi()+model_.P(DVector<double>::Constant(1,lambda)));
            invD_ = chol_.matrixL().solve(DMatrix<double>::Identity(model_.n_basis(),model_.n_basis()));
        }
        // compute SVD of X*\Psi*(D^{-1})^\top
        if constexpr (is_rand_svd<SVDType>{}){
            svd_.compute(train_set.select(model_.X())*model_.Psi()*invD_.transpose(),rank_);
        } else{
            svd_.compute(train_set.select(model_.X())*model_.Psi()*invD_.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        const auto end{std::chrono::steady_clock::now()};
        svd_time_ = (std::chrono::duration<double>{end - start}).count();
        return;
    }
    //gcv score (has to be preceded by a call to compute())
    double gcv(){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();
        double gcv_score = model_.X().cols()/std::pow(model_.X().cols()-S_m.trace(),2)*
                           (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm();
        return gcv_score;
    }
    //ocv score (has to be preceded by a call to compute())
    double ocv(){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();

        DVector<double> pred_err = (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).colwise().squaredNorm();
        double ocv_score = (
                pred_err.array()
                / (1 - S_m.diagonal().array()).square()
        ).sum();
        return ocv_score;
    }
    //test the reconstruction on a different portion of the data (has to be preceded by a call to compute())
    double reconstruction_error(const fdapde::core::BinaryMatrix<Dynamic,Dynamic>& test_set){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();

        return test_set.select(
                model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)
        ).squaredNorm() / (test_set.rows() * test_set.cols());
    }
    //getters
    const DMatrix<double> scores() const { return svd_.matrixU().leftCols(rank_); }
    const DMatrix<double> loadings() const { return (svd_.singularValues().head(rank_).asDiagonal()*svd_.matrixV().leftCols(rank_).transpose()*invD_).transpose(); }
    //time report
    double factorization_time() const{ return factorization_time_;}
    double fit_time() const{ return svd_time_;}
};

//Sparse monolithic solver
template<typename ModelType, typename SVDType=RSVD<DMatrix<double>>>
class SparseMonoFPCASolver{
private:
    const ModelType &model_;
    int rank_;
    double lambda_;
    //Factorization of the smoothing matrix
    Eigen::LLT<DMatrix<double>> chol_; //cholesky of Psi^T*Psi
    DMatrix<double> invD_; //factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
    //Solutions to the fPCA problem
    SVDType svd_; //here we use a randomized algorithm
    //Time report
    double factorization_time_;
    double svd_time_;

public:
    SparseMonoFPCASolver(const ModelType &model, double lambda) : model_(model), lambda_(lambda){
        const auto start{std::chrono::steady_clock::now()};

        SparseBlockMatrix<double, 2, 2> C(
                -model.lambda_D()*model.R0(), model.lambda_D()*model.R1(),
                model.lambda_D()*model.R1(), model.Psi().transpose()*model.Psi()
        );

        //Working version: natural ordering (problem: potential fill-in issues)
        Eigen::SimplicialLDLT<SpMatrix<double>,Eigen::Lower,Eigen::NaturalOrdering<typename SpMatrix<double>::StorageIndex>> chol;
        chol.compute(C);

        int n_nodes = model.n_basis();

        invD_ = chol.matrixL().bottomRightCorner(n_nodes,n_nodes).triangularView<Eigen::Lower>().solve(DMatrix<double>::Identity(n_nodes,n_nodes));
        invD_ = chol.vectorD().tail(n_nodes).unaryExpr([](double x){ return x > 0? 1/std::sqrt(x) : 0;}).asDiagonal()*invD_;

        const auto end{std::chrono::steady_clock::now()};

        factorization_time_ = (std::chrono::duration<double>{end - start}).count();
    }
    void compute(const fdapde::core::BinaryMatrix<Dynamic,Dynamic>& train_set, int rank, double lambda){
        const auto start{std::chrono::steady_clock::now()};
        rank_ = rank;
        if(lambda_ != lambda){
            lambda_ = lambda;

            SparseBlockMatrix<double, 2, 2> C(
                    -lambda_*model_.R0(), lambda_*model_.R1(),
                    lambda_*model_.R1(), model_.Psi().transpose()*model_.Psi()
            );
            //Working version: natural ordering (problem: potential fill-in issues)
            Eigen::SimplicialLDLT<SpMatrix<double>,Eigen::Lower,Eigen::NaturalOrdering<typename SpMatrix<double>::StorageIndex>> chol;
            chol.compute(C);

            int n_nodes = model_.n_basis();
            invD_ = chol.matrixL().bottomRightCorner(n_nodes,n_nodes).triangularView<Eigen::Lower>().solve(DMatrix<double>::Identity(n_nodes,n_nodes));
            invD_ = chol.vectorD().tail(n_nodes).unaryExpr([](double x){ return x > 0? 1/std::sqrt(x) : 0;}).asDiagonal()*invD_;
        }
        // compute SVD of X*\Psi*(D^{-1})^\top
        if constexpr (is_rand_svd<SVDType>{}){
            svd_.compute(train_set.select(model_.X())*model_.Psi()*invD_.transpose(),rank_);
        } else{
            svd_.compute(train_set.select(model_.X())*model_.Psi()*invD_.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        const auto end{std::chrono::steady_clock::now()};
        svd_time_ = (std::chrono::duration<double>{end - start}).count();
        return;
    }
    //gcv score (has to be preceded by a call to compute())
    double gcv(){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();
        double gcv_score = model_.X().cols()/std::pow(model_.X().cols()-S_m.trace(),2)*
                           (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).squaredNorm();
        return gcv_score;
    }
    //ocv score (has to be preceded by a call to compute())
    double ocv(){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();

        DVector<double> pred_err = (model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)).colwise().squaredNorm();
        double ocv_score = (
                pred_err.array()
                / (1 - S_m.diagonal().array()).square()
        ).sum();
        return ocv_score;
    }
    //test the reconstruction on a different portion of the data (has to be preceded by a call to compute())
    double reconstruction_error(const fdapde::core::BinaryMatrix<Dynamic,Dynamic>& test_set){
        DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV().leftCols(rank_);
        S_m = S_m*S_m.transpose();

        return test_set.select(
                model_.X()*(DMatrix<double>::Identity(model_.X().cols(), model_.X().cols()) - S_m)
        ).squaredNorm() / (test_set.rows() * test_set.cols());
    }
    //getters
    const DMatrix<double> scores() const { return svd_.matrixU().leftCols(rank_); }
    const DMatrix<double> loadings() const { return (svd_.singularValues().head(rank_).asDiagonal()*svd_.matrixV().leftCols(rank_).transpose()*invD_).transpose(); }
    //time report
    double factorization_time() const{ return factorization_time_;}
    double fit_time() const{ return svd_time_;}
};