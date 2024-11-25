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

#ifndef __REGULARIZED_SVD_H__
#define __REGULARIZED_SVD_H__

#include <fdaPDE/utils.h>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>

#include "fdaPDE/core/fdaPDE/linear_algebra.h"
using fdapde::core::RSVD;

#include "../../calibration/kfold_cv.h"
#include "../../calibration/symbols.h"
using fdapde::calibration::Calibration;

#include "../model_traits.h"
#include "power_iteration.h"
#include "../../core/fdaPDE/optimization/grid.h"

#include <chrono>

namespace fdapde {

//new policies to exploit sparsity
struct monolithic_fpsai{};
struct monolithic_spchol{};

namespace models {

// Let X be a data matrix made of noisy and discrete measurements of smooth functions sampled from a random field
// \mathcal{X}. RegularizedSVD implements the computation of a low-rank approximation of X using some regularizing term
template <typename SolutionPolicy_> class RegularizedSVD;

// Finds a low-rank approximation of X while penalizing for the eigenfunctions of \mathcal{X} by sequentially solving
// \argmin_{s,f} \norm_F{X - s^\top*f}^2 + (s^\top*s)*P_{\lambda}(f), up to a desired rank
template<>
class RegularizedSVD<sequential> {
   private:
    Calibration calibration_;    // PC function's smoothing parameter selection strategy
    int n_folds_ = 10;   // for a kcv calibration strategy, the number of folds
    DMatrix<double> lambda_grid_;
    // power iteration parameters
    double tolerance_ = 1e-6;   // relative tolerance between Jnew and Jold, used as stopping criterion
    int max_iter_ = 20;         // maximum number of allowed iterations
    int seed_ = fdapde::random_seed;

    // problem solution
    DMatrix<double> loadings_;        // PC functions' expansion coefficients
    DVector<double> loadings_norm_;   // L^2 norm of estimated fields
    DMatrix<double> scores_;
    std::vector<DVector<double>> selected_lambdas_;   // vector of smoothing parameters selected for each component

    // rank-one-stepper API: this iterator allows to range over each component of X, while triggering the computation
    // only when required
    template <typename ModelType>
    struct rsvd_iterator {
       private:
        friend RegularizedSVD;
        RegularizedSVD* rsvd_;
        int index_;                               // current rank
        DMatrix<double> X_;                       // deflated data
        PowerIteration<ModelType> solver_;        // rank-one step solver
        RSVD<DMatrix<double>> svd_;   // Randomized Singular Value Decomposition (not regularized)(truncated)
        ModelType& model_;

        // rank-one step: \argmin_{s,f} \norm_F{X - s^\top*f}^2 + (s^\top*s)*P_{\lambda}(f). calibration of \lambda
        // dispatched to desired strategy
        void rank_one_step() {
            DVector<double> f0 = svd_.matrixV().col(index_);
            // select optimal smoothing level according to requested calibration strategy
            DVector<double> optimal_lambda;
            switch (rsvd_->calibration_) {
            case Calibration::off: {
                // find vectors s,f minimizing \norm_F{Y - s^T*f}^2 + (s^T*s)*P(f) fixed \lambda
                optimal_lambda = model_.lambda();
            } break;
            case Calibration::gcv: {
                // select \lambda minimizing the GCV index
                ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                    solver_.compute(X_, lambda, f0);
                    return solver_.gcv();   // return GCV index at convergence
                });
		optimal_lambda = core::Grid<Dynamic> {}.optimize(gcv, rsvd_->lambda_grid_);
            } break;
            case Calibration::kcv: {
                // select \lambda minimizing the reconstruction error in cross-validation
                auto cv_score = [&](
                                  const DVector<double>& lambda, const core::BinaryVector<Dynamic>& train_set,
                                  const core::BinaryVector<Dynamic>& test_set) -> double {
                    solver_.compute(train_set.repeat(1, X_.cols()).select(X_), lambda, f0);   // fit on train set
                    // reconstruction error on test set: \norm{X_test * (I - fn*fn^\top/J)}_F/n_test, with
                    // J = \norm{f_n}_2^2 + f^\top*P(\lambda)*f (PS: the division of f^\top*P(\lambda)*f by
                    // \norm{f}_{L^2} is necessary to obtain J as expected)
                    return (test_set.repeat(1, X_.cols()).select(X_) *
                            (DMatrix<double>::Identity(X_.cols(), X_.cols()) -
                             solver_.fn() * solver_.fn().transpose() /
                               (solver_.fn().squaredNorm() + solver_.ftPf(lambda) / solver_.f_squaredNorm())))
                             .squaredNorm() /
                           test_set.count() * X_.cols();
                };
                optimal_lambda =
                  calibration::KCV {rsvd_->n_folds_, rsvd_->seed_}.fit(model_, rsvd_->lambda_grid_, cv_score);
            } break;
            }
            solver_.compute(X_, optimal_lambda, f0);
            X_ -= solver_.s() * solver_.fn().transpose() * solver_.f_norm();   // X <- X - s*f_n^\top (deflation step)
            rsvd_->selected_lambdas_.push_back(optimal_lambda);                // store optimal smoothing level
            return;
        }
       public:
        // constructor
        rsvd_iterator(RegularizedSVD* rsvd, int index, int rank, const DMatrix<double>& X, ModelType& model) :
            rsvd_(rsvd), index_(index), X_(X), solver_(model, rsvd->tolerance_, rsvd->max_iter_, rsvd->seed_),
            model_(model) {
            // first guess of PCs set to a multivariate PCA (SVD)
            const auto start_svd{std::chrono::steady_clock::now()};
            svd_.compute(X_,rank);
            const auto end_svd{std::chrono::steady_clock::now()};
            std::ofstream t_svd("results/time_svd.csv");
            t_svd  << (std::chrono::duration<double>{end_svd - start_svd}).count() << std::endl;
            t_svd.close();

            solver_.init();   // initialize power iteration solver
        };
        rsvd_iterator& operator++() {
            ++index_;
            return *this;
        }
        // iterate until desired rank not reached
        bool operator!=(int rank) {
            fdapde_assert(rank > 0);
            if (index_ != rank) { rank_one_step(); }   // trigger computation of next component only if not ended
            return index_ != rank;
        }
        // getters
        const DVector<double>& scores() const { return solver_.s(); }
        const DVector<double>& loading() const { return solver_.f(); }
        double norm() const { return solver_.f_norm(); }
        const DVector<double>& lambda() const { return rsvd_->selected_lambdas_.back(); }
    };
   public:
    // constructors
    RegularizedSVD(Calibration c) : calibration_(c){};
    RegularizedSVD() : RegularizedSVD(Calibration::off){};

    // sequentially solves \argmin_{s,f} \norm_F{X - s^\top*f}^2 + (s^\top*s)*P_{\lambda}(f), up to the specified rank,
    // selecting the level of smoothing of the component according to the desired strategy
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {
        // preallocate space
        loadings_.resize(model.n_basis(), rank);
        scores_.resize(X.rows(), rank);
        loadings_norm_.resize(rank);
        int i = 0;
        for (auto it = rank_one_stepper(X, rank, model); it != rank; ++it, ++i) {
            loadings_.col(i) = it.loading();
            scores_.col(i) = it.scores() * it.norm();
            loadings_norm_[i] = it.norm();
        }
    }
    // iterator support
    template <typename ModelType>
    rsvd_iterator<ModelType> rank_one_stepper(const DMatrix<double>& X, int rank, ModelType&& model) {
        return rsvd_iterator<ModelType>(this, 0, rank, X, model);
    }
    // getters
    const DMatrix<double>& scores() const { return scores_; }
    const DMatrix<double>& loadings() const { return loadings_; }
    const DVector<double>& loadings_norm() const { return loadings_norm_; }
    const std::vector<DVector<double>>& selected_lambdas() const { return selected_lambdas_; }
    const DMatrix<double>& lambda_grid() const { return lambda_grid_; }
    Calibration calibration() const { return calibration_; }
    const DVector<double>& lambda() const { return selected_lambdas_.back(); }

    // setters
    void set_tolerance(double tolerance) { tolerance_ = tolerance; }
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
    void set_seed(int seed) { seed_ = seed; }
    RegularizedSVD& set_lambda(const DMatrix<double>& lambda_grid) {
        fdapde_assert(calibration_ != Calibration::off);
        lambda_grid_ = lambda_grid;
        return *this;
    }
    RegularizedSVD& set_nfolds(int n_folds) {
        fdapde_assert(calibration_ == Calibration::kcv);
        n_folds_ = n_folds;
        return *this;
    }
};

// finds a rank r matrix U minimizing \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top]
template<>
class RegularizedSVD<monolithic> {
private:
    Calibration calibration_;    // PC function's smoothing parameter selection strategy
    int n_folds_ = 10;   // for a kcv calibration strategy, the number of folds
    DMatrix<double> lambda_grid_;
    int seed_ = fdapde::random_seed;

    template <typename ModelType>
    struct internal_solver{
    private:
        const DMatrix<double> &X_;
        ModelType& model_;
        double lambda_;
        //Solution to the generalized eigenvalue problem
        Eigen::SelfAdjointEigenSolver<DMatrix<double>> evd_;
        Eigen::SimplicialLLT<SpMatrix<double>> chol_; //cholesky of Psi^T*Psi
        DMatrix<double> invL_; // inverse of the cholesky factor
        //Solution to the fpca problem
        RSVD<DMatrix<double>> svd_; //here we use a randomized algorithm
        DMatrix<double> invD_; //factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
    public:
        internal_solver(const DMatrix<double> &X, ModelType& model) : X_(X), model_(model) {
            //Genelarized eigenvalue problem: P*V = (Psi^T*Psi)*V*Lambda
            chol_.compute(model.Psi().transpose()*model.Psi());
            invL_ = chol_.matrixL().solve(DMatrix<double>::Identity(model.n_basis(),model.n_basis()))*chol_.permutationP().toDenseMatrix().cast<double>();
            evd_.compute(invL_*model.P(DVector<double>::Ones(1))*invL_.transpose());
        };
        void compute(const core::BinaryVector<Dynamic>& train_set, int rank, double lambda){
            lambda_ = lambda;
            // assemble the factorization of (Psi^T*Psi+lambda P)^(-1) for the given lambda
            DMatrix<double> V = invL_.transpose()*evd_.eigenvectors();
            invD_ = (DVector<double>::Ones(model_.n_basis())+lambda_*evd_.eigenvalues()).unaryExpr([](double x){ return 1/std::sqrt(x);}).asDiagonal()*V.transpose();
            // compute SVD of X*\Psi*(D^{-1})^\top
            svd_.compute(train_set.repeat(1, X_.cols()).select(X_)*model_.Psi()*invD_.transpose(),rank);
            return;
        }
        //gcv score
        double gcv(){
            DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV();
            //DMatrix<double> tmp = svd_.matrixV().transpose()*(DMatrix<double>::Identity(model_.n_basis(),model_.n_basis()) - invD_*(model_.Psi().transpose()*model_.Psi())*invD_.transpose())*svd_.matrixV();
            S_m = S_m*S_m.transpose();

            double gcv_score = X_.cols()/std::pow(X_.cols()-S_m.trace(),2)*
                               (X_*(DMatrix<double>::Identity(X_.cols(), X_.cols()) - S_m)).squaredNorm();

            //Computation of the GCV index
            return gcv_score;
        }
        double reconstruction_error(const core::BinaryVector<Dynamic>& test_set){
            DMatrix<double> S_m = model_.Psi()*invD_.transpose()*svd_.matrixV();
            S_m = S_m*S_m.transpose();
            DMatrix<double> X_test = test_set.repeat(1, X_.cols()).select(X_);

            return (X_test*
                    (DMatrix<double>::Identity(X_test.cols(), X_test.cols()) - S_m)).squaredNorm() /
                   (X_test.rows() * X_test.cols());
        }
        const DMatrix<double> scores() const { return svd_.matrixU(); }
        const DMatrix<double> loadings() const { return (svd_.singularValues().asDiagonal()*svd_.matrixV().transpose()*invD_).transpose(); }
    };

public:
    // constructors
    RegularizedSVD(Calibration c) : calibration_(c){};
    RegularizedSVD() : RegularizedSVD(Calibration::off){};

    // solves \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top] retaining the first rank components
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {
        //Calibration
        DVector<double> optimal_lambda;
        auto mono_solver = internal_solver(X,model);

        switch (calibration_) {
            case Calibration::off: {
                optimal_lambda = model.lambda();
            } break;
            case Calibration::gcv: {
                // select \lambda minimizing the GCV index
                ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                    mono_solver.compute(core::BinaryVector<Dynamic>::Ones(X.rows()),rank,lambda(0));
                    return mono_solver.gcv();
                });
                optimal_lambda = core::Grid<Dynamic> {}.optimize(gcv, lambda_grid_);
            } break;
            case Calibration::kcv:{
                // select \lambda minimizing the reconstruction error in cross-validation
                auto cv_score = [&](
                        const DVector<double>& lambda,
                        const core::BinaryVector<Dynamic>& train_set,
                        const core::BinaryVector<Dynamic>& test_set) -> double {
                    //fitting on the training set
                    mono_solver.compute(train_set,rank,lambda(0));
                    //evaluate error on the test set
                    return mono_solver.reconstruction_error(test_set);
                };
                optimal_lambda =
                        calibration::KCV{n_folds_, seed_, false}.fit(model, lambda_grid_, cv_score);
            } break;
        }
        selected_lambdas_.push_back(optimal_lambda);

        //Run using the optimal lambda
        model.set_lambda(selected_lambdas_.back());
        //train on all the data
        mono_solver.compute(core::BinaryVector<Dynamic>::Ones(X.rows()),rank,optimal_lambda(0));

        // store results
        scores_ = mono_solver.scores();
        loadings_ = mono_solver.loadings();
        loadings_norm_.resize(rank);
        for (int i = 0; i < rank; ++i) {
            loadings_norm_[i] = std::sqrt(loadings_.col(i).dot(model.R0() * loadings_.col(i)));   // L^2 norm
            loadings_.col(i) = loadings_.col(i) / loadings_norm_[i];
        }
        scores_ = scores_.array().rowwise() * loadings_norm_.transpose().array();
        return;
    }
    // getters
    const DMatrix<double>& scores() const { return scores_; }
    const DMatrix<double>& loadings() const { return loadings_; }
    const DVector<double>& loadings_norm() const { return loadings_norm_; }
    const DMatrix<double>& lambda_grid() const { return lambda_grid_; }
    Calibration calibration() const { return calibration_; }
    const std::vector<DVector<double>>& selected_lambdas() const { return selected_lambdas_; }

    //setters
    RegularizedSVD& set_lambda(const DVector<double>& lambda_grid) {
        fdapde_assert(calibration_ != Calibration::off);
        lambda_grid_ = lambda_grid;
        return *this;
    }

   private:
    // let E*\Sigma*F^\top the reduced (rank r) SVD of X*\Psi*(D^{1})^\top, with D^{-1} the inverse of the cholesky
    // factor of \Psi^\top * \Psi + P(\lambda), then
    DMatrix<double> scores_;          // matrix E in the reduced SVD of X*\Psi*(D^{-1})^\top
    DMatrix<double> loadings_;        // \Sigma*F^\top*D^{-1} (PC functions expansion coefficients, L^2 normalized)
    DVector<double> loadings_norm_;   // L^2 norm of estimated fields
    std::vector<DVector<double>> selected_lambdas_;
};

//Exploiting sparsity
template<>
class RegularizedSVD<monolithic_spchol> {
public:
    // constructor
    RegularizedSVD(){};

    // solves \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top] retaining the first rank components
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {

        const auto start_chol{std::chrono::steady_clock::now()};

        SparseBlockMatrix<double, 2, 2> C(
                -model.lambda_D()*model.R0(), model.lambda_D()*model.R1(),
                model.lambda_D()*model.R1(), model.Psi().transpose()*model.Psi()
        );

        //Working version: natural ordering (problem: potential fill-in issues)
        Eigen::SimplicialLDLT<SpMatrix<double>,Eigen::Lower,Eigen::NaturalOrdering<typename SpMatrix<double>::StorageIndex>> chol;
        chol.compute(C);

        int n_nodes = model.n_basis();

        DMatrix<double> invD = chol.matrixL().bottomRightCorner(n_nodes,n_nodes).triangularView<Eigen::Lower>().solve(DMatrix<double>::Identity(n_nodes,n_nodes));
        invD = chol.vectorD().tail(n_nodes).unaryExpr([](double x){ return x > 0? 1/std::sqrt(x) : 0;}).asDiagonal()*invD;

        //Test: using AMD reordering mess up with the blocked diagonal and prevents us to factorized the Schur complement
        /*
        Eigen::SimplicialLDLT<SpMatrix<double>> chol_amd;
        chol_amd.compute(C);

        DMatrix<double> selector1 = chol_amd.permutationP().toDenseMatrix().cast<double>().leftCols(n_nodes);
        DMatrix<double> selector2 = chol_amd.permutationP().toDenseMatrix().cast<double>().rightCols(n_nodes);

        DMatrix<double> invD = selector2.transpose()*chol_amd.matrixL().solve(DMatrix<double>::Identity(C.rows(),C.cols()))*selector2;


        DMatrix<double> diag = (chol_amd.permutationPinv()*chol_amd.vectorD()).asDiagonal();

        diag = selector1.transpose()*selector2*diag.topLeftCorner(n_nodes,n_nodes)*selector2.transpose()*selector1 + diag.bottomRightCorner(n_nodes,n_nodes);

        std::cout << diag.diagonal() << std::endl;

        diag = diag.unaryExpr([](double x){return x > 0? 1/std::sqrt(x) : 0;});
        std::cout << diag.diagonal() << std::endl;

        invD = diag*invD;
         */

        const auto end_chol{std::chrono::steady_clock::now()};
        std::ofstream t_chol("results/time_other.csv");
        t_chol << (std::chrono::duration<double>{end_chol - start_chol}).count() << std::endl;
        t_chol.close();

        const auto start_svd{std::chrono::steady_clock::now()};
        // compute SVD of X*\Psi*(D^{-1})^\top
        RSVD<DMatrix<double>> tr_svd(X * model.Psi() * invD.transpose(),rank);
        const auto end_svd{std::chrono::steady_clock::now()};
        std::ofstream t_svd("results/time_svd.csv");
        t_svd  << (std::chrono::duration<double>{end_svd - start_svd}).count() << std::endl;
        t_svd.close();

        // store results
        scores_ = tr_svd.matrixU().leftCols(rank);
        loadings_ =
                (tr_svd.singularValues().head(rank).asDiagonal() * tr_svd.matrixV().leftCols(rank).transpose()*invD).transpose();
        loadings_norm_.resize(rank);
        for (int i = 0; i < rank; ++i) {
            loadings_norm_[i] = std::sqrt(loadings_.col(i).dot(model.R0() * loadings_.col(i)));   // L^2 norm
            loadings_.col(i) = loadings_.col(i) / loadings_norm_[i];
        }
        scores_ = scores_.array().rowwise() * loadings_norm_.transpose().array();
        return;
    }
    // getters
    const DMatrix<double>& scores() const { return scores_; }
    const DMatrix<double>& loadings() const { return loadings_; }

    const DVector<double>& loadings_norm() const { return loadings_norm_; }
    Calibration calibration() const { return Calibration::off; }   // calibration is implicitly off
    const std::vector<DVector<double>>& selected_lambdas() const { return std::vector<DVector<double>>(); }

private:
    // let E*\Sigma*F^\top the reduced (rank r) SVD of X*\Psi*(D^{1})^\top, with D^{-1} the inverse of the cholesky
    // factor of \Psi^\top * \Psi + P(\lambda), then
    DMatrix<double> scores_;          // matrix E in the reduced SVD of X*\Psi*(D^{-1})^\top
    DMatrix<double> loadings_;        // \Sigma*F^\top*D^{-1} (PC functions expansion coefficients, L^2 normalized)
    DVector<double> loadings_norm_;   // L^2 norm of estimated fields
};


}   // namespace models
}   // namespace fdapde

#endif   // __REGULARIZED_SVD_H__
