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
using fdapde::core::is_rand_svd;

#include "../../calibration/kfold_cv.h"
#include "../../calibration/symbols.h"
using fdapde::calibration::Calibration;
using fdapde::calibration::KCVPolicy;

#include "../model_traits.h"
#include "power_iteration.h"
#include "monolithic_solver.h"
#include "../../core/fdaPDE/optimization/grid.h"

//test
#include <chrono>
#include "../../../test/src/utils/utils.h"
using fdapde::testing::write_csv;

namespace fdapde {


namespace models{

// Let X be a data matrix made of noisy and discrete measurements of smooth functions sampled from a random field
// \mathcal{X}. RegularizedSVD implements the computation of a low-rank approximation of X using some regularizing term
template <typename SolutionPolicy_, typename SVDType> class RegularizedSVD;

// Finds a low-rank approximation of X while penalizing for the eigenfunctions of \mathcal{X} by sequentially solving
// \argmin_{s,f} \norm_F{X - s^\top*f}^2 + (s^\top*s)*P_{\lambda}(f), up to a desired rank
template<typename SVDType>
class RegularizedSVD<sequential,SVDType> {
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
        SVDType svd_;   //Randomized Singular Value Decomposition (not regularized)(truncated)
        ModelType& model_;

        // rank-one step: \argmin_{s,f} \norm_F{X - s^\top*f}^2 + (s^\top*s)*P_{\lambda}(f). calibration of \lambda
        // dispatched to desired strategy
        void rank_one_step() {
            DVector<double> f0 = svd_.matrixV().col(index_);
            // select optimal smoothing level according to requested calibration strategy
            DVector<double> optimal_lambda;
            const auto start{std::chrono::steady_clock::now()};
            switch (rsvd_->calibration_){
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
            const auto end{std::chrono::steady_clock::now()};

            DMatrix<double> calibration_time = DMatrix<double>::Constant(1,1,(std::chrono::duration<double>{end - start}).count());
            write_csv("tmp/calibration_time.csv",calibration_time);

            const auto start_fit{std::chrono::steady_clock::now()};
            solver_.compute(X_, optimal_lambda, f0);
            X_ -= solver_.s() * solver_.fn().transpose() * solver_.f_norm();   // X <- X - s*f_n^\top (deflation step)
            rsvd_->selected_lambdas_.push_back(optimal_lambda);                // store optimal smoothing level
            const auto end_fit{std::chrono::steady_clock::now()};

            DMatrix<double> fit_time = DMatrix<double>::Constant(1,1,(std::chrono::duration<double>{end_fit - start_fit}).count());
            write_csv("tmp/fit_time.csv",fit_time);

            return;
        }
       public:
        // constructor
        rsvd_iterator(RegularizedSVD* rsvd, int index, int rank, const DMatrix<double>& X, ModelType& model) :
            rsvd_(rsvd), index_(index), X_(X), solver_(model, rsvd->tolerance_, rsvd->max_iter_, rsvd->seed_),
            model_(model) {
            // first guess of PCs set to a multivariate PCA (SVD)
            const auto start{std::chrono::steady_clock::now()};
            if constexpr (is_rand_svd<SVDType>{}){
                svd_.compute(X_,rank);
            } else{
                svd_.compute(X_,Eigen::ComputeThinU | Eigen::ComputeThinV);
            }
            const auto end{std::chrono::steady_clock::now()};
            std::ofstream svd_time("tmp/svd_time.csv");
            svd_time  << '"' << (std::chrono::duration<double>{end - start}).count() << '"' << std::endl;
            svd_time.close();

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
template<typename SVDType>
class RegularizedSVD<monolithic,SVDType> {
private:
    Calibration calibration_;    // PC function's smoothing parameter selection strategy
    int n_folds_ = 10;   // for a kcv calibration strategy, the number of folds
    DMatrix<double> lambda_grid_;
    int seed_ = fdapde::random_seed;
public:
    // constructors
    RegularizedSVD(Calibration c) : calibration_(c){};
    RegularizedSVD() : RegularizedSVD(Calibration::off){};

    // solves \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top] retaining the first rank components
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {
        //Calibration
        DVector<double> optimal_lambda;
        if (calibration_ == Calibration::off){
            optimal_lambda = model.lambda();
            Eigen::LLT<DMatrix<double>> chol(model.Psi().transpose()*model.Psi()+model.P(optimal_lambda));
            DMatrix<double> invD = chol.matrixL().solve(DMatrix<double>::Identity(model.n_basis(),model.n_basis()));

            SVDType svd;
            svd.compute(X*model.Psi()*invD.transpose(),rank);
            if constexpr (is_rand_svd<SVDType>{}){
                svd.compute(X*model.Psi()*invD.transpose(),rank);
            } else{
                svd.compute(X*model.Psi()*invD.transpose(),Eigen::ComputeThinU | Eigen::ComputeThinV);
            }
            // store results
            scores_ = svd.matrixU().leftCols(rank);
            loadings_ = (svd.singularValues().head(rank).asDiagonal()*svd.matrixV().leftCols(rank).transpose()*invD).transpose();
            loadings_norm_.resize(rank);
            for (int i = 0; i < rank; ++i) {
                loadings_norm_[i] = std::sqrt(loadings_.col(i).dot(model.R0() * loadings_.col(i)));   // L^2 norm
                loadings_.col(i) = loadings_.col(i) / loadings_norm_[i];
            }
            scores_ = scores_.array().rowwise() * loadings_norm_.transpose().array();
            return; //exit
        }
        //if the calibration is activated it is better to use the internal solver to avoid recomputing the cholesky for every lambda
        const auto start_calibration{std::chrono::steady_clock::now()};
        MonolithicSolver<ModelType,SVDType> mono_solver(model, seed_);
        switch (calibration_) {
            case Calibration::gcv: {
                // select \lambda minimizing the GCV index
                ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                    mono_solver.compute(X,rank,lambda(0));
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
                    mono_solver.compute(train_set.repeat(1,X.cols()).select(X),rank,lambda(0));
                    //evaluate the error on the test set
                    return mono_solver.reconstruction_error(test_set.repeat(1,X.cols()).select(X));
                };
                optimal_lambda =
                        calibration::KCV{n_folds_, seed_}.fit(model, lambda_grid_, cv_score);
            } break;
        }
        const auto end_calibration{std::chrono::steady_clock::now()};
        DMatrix<double> calibration_time = DMatrix<double>::Constant(1,1,(std::chrono::duration<double>{end_calibration - start_calibration}).count());
        write_csv("tmp/calibration_time.csv",calibration_time);

        selected_lambdas_.push_back(optimal_lambda);
        //Run using the optimal lambda
        model.set_lambda(selected_lambdas_.back());

        //train on all the data
        const auto start_fit{std::chrono::steady_clock::now()};
        mono_solver.compute(X,rank,optimal_lambda(0));
        const auto end_fit{std::chrono::steady_clock::now()};

        DMatrix<double> fit_time = DMatrix<double>::Constant(1,1,(std::chrono::duration<double>{end_fit - start_fit}).count());
        write_csv("tmp/fit_time.csv",fit_time);

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
    RegularizedSVD& set_nfolds(int n_folds) {
        //fdapde_assert(calibration_ == Calibration::kcv);
        n_folds_ = n_folds;
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


struct miss_monolithic{};

template<typename SVDType>
class RegularizedSVD<miss_monolithic,SVDType> {
private:
    double lambda_; //the algorithm works at fixed lambda
    int seed_ = fdapde::random_seed;
    double tol_;
public:
    // constructors
    RegularizedSVD() = default;

    // solves \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top] retaining the first rank components
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {
        lambda_ = model.lambda()(0);

        DMatrix<bool> W = !X.array().isNaN();
        DMatrix<double> U_old = DMatrix<double>::Zero(X.rows(),model.n_basis());
        DMatrix<double> U = DMatrix<double>::Zero(X.rows(),model.n_basis());

        MonolithicSolver<ModelType,SVDType> solver(model, seed_);
        solver.init(); //compute the factorization just once
        double rec_err = tol_+1;
        for(int k = 1; k <= rank; ++k){
            //Majorization-Minimization scheme
            while(rec_err > tol_){
                //fit on the imputed data
                solver.compute(W.select(X,0)+(!W.array()).select(U_old*model.Psi().transpose(),0),
                               k,lambda_);
                //error
                U_old = U;
                U = solver.scores()*solver.loadings().transpose();
                rec_err = ((U-U_old)*model.Psi().transpose()).norm(); //updating the reconstruction
            }
            rec_err = tol_+1;
        }
        // store results
        scores_ = solver.scores();
        loadings_ = solver.loadings();
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
    //setters
    RegularizedSVD& set_lambda(DVector<double> lambda) {
        lambda_ = lambda(0);
        return *this;
    }
private:
    // let E*\Sigma*F^\top the reduced (rank r) SVD of X*\Psi*(D^{1})^\top, with D^{-1} the inverse of the cholesky
    // factor of \Psi^\top * \Psi + P(\lambda), then
    DMatrix<double> scores_;          // matrix E in the reduced SVD of X*\Psi*(D^{-1})^\top
    DMatrix<double> loadings_;        // \Sigma*F^\top*D^{-1} (PC functions expansion coefficients, L^2 normalized)
    DVector<double> loadings_norm_;   // L^2 norm of estimated fields
};


struct miss_sequential{};
template<typename SVDType>
class RegularizedSVD<miss_sequential,SVDType>{
private:
    DVector<double> lambda_;
    // power iteration parameters
    double tolerance_ = 1e-6;   // relative tolerance between Jnew and Jold, used as stopping criterion
    int max_iter_ = 20;         // maximum number of allowed iterations
    int seed_ = fdapde::random_seed;

public:
    // constructors
    RegularizedSVD() = default;

    // solves \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top] retaining the first rank components
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {
        lambda_ = model.lambda();
        //init loadings and scores dimensions
        loadings_.resize(model.n_basis(), rank);
        scores_.resize(X.rows(), rank);
        loadings_norm_.resize(rank);
        //MM-scheme init
        DMatrix<bool> W = !X.array().isNaN();
        DMatrix<double> U_old = DMatrix<double>::Zero(X.rows(),model.n_basis());
        DMatrix<double> U = DMatrix<double>::Zero(X.rows(),model.n_basis());
        //Init the solver
        PowerIteration<ModelType> solver(model, tolerance_,max_iter_,seed_);
        solver.init(); //compute the factorization just once
        SVDType svd;
        double rec_err = tolerance_+1;
        for(int k = 1; k <= rank; ++k){
            //Majorization-Minimization scheme
            while(rec_err > tolerance_){
                //init
                DMatrix<double> X_imputed = W.select(X,0)+(!W.array()).select(U_old*model.Psi().transpose(),0);
                if constexpr (is_rand_svd<SVDType>{}){
                    svd.compute(X_imputed,rank);
                } else{
                    svd.compute(X_imputed,Eigen::ComputeThinU | Eigen::ComputeThinV);
                }
                //sequential estimates
                for(int index=0; index<k; index++){
                    //fit on the imputed data
                    solver.compute(X_imputed,lambda_,svd.matrixV().col(index));
                    //deflation
                    X_imputed -= solver.s() * solver.fn().transpose() * solver.f_norm();
                    loadings_.col(index) = solver.f();
                    scores_.col(index) = solver.s() * solver.f_norm();
                    loadings_norm_[index] = solver.f_norm();
                }
                //error
                U_old = U;
                U = scores_.leftCols(k)*loadings_.leftCols(k).transpose();
                rec_err = ((U-U_old)*model.Psi().transpose()).norm(); //updating the reconstruction
            }
            rec_err = tolerance_+1;
        }
        return;
    }
    // getters
    const DMatrix<double>& scores() const { return scores_; }
    const DMatrix<double>& loadings() const { return loadings_; }
    const DVector<double>& loadings_norm() const { return loadings_norm_; }
    //setters
    RegularizedSVD& set_lambda(DVector<double> lambda) {
        lambda_ = lambda;
        return *this;
    }
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
