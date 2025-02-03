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
template <typename SolutionPolicy_, typename SVDType_, bool MissingData_=false> class RegularizedSVD;

// Finds a low-rank approximation of X while penalizing for the eigenfunctions of \mathcal{X} by sequentially solving
// \argmin_{s,f} \norm_F{X - s^\top*f}^2 + (s^\top*s)*P_{\lambda}(f), up to a desired rank
template<typename SVDType_>
class RegularizedSVD<sequential,SVDType_> {
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
        SVDType_ svd_;   //Randomized Singular Value Decomposition (not regularized)(truncated)
        ModelType& model_;

        // rank-one step: \argmin_{s,f} \norm_F{X - s^\top*f}^2 + (s^\top*s)*P_{\lambda}(f). calibration of \lambda
        // dispatched to desired strategy
        void rank_one_step() {
            DVector<double> f0 = svd_.matrixV().col(index_);
            // select optimal smoothing level according to requested calibration strategy
            DVector<double> optimal_lambda;

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
            const auto start{std::chrono::steady_clock::now()};
            if constexpr (is_rand_svd<SVDType_>{}){
                svd_.compute(X_,rank);
            } else{
                svd_.compute(X_,Eigen::ComputeThinU | Eigen::ComputeThinV);
            }
            const auto end{std::chrono::steady_clock::now()};
            std::ofstream svd_time("results/svd_time.csv");
            svd_time  << (std::chrono::duration<double>{end - start}).count() << std::endl;
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
template<typename SVDType_>
class RegularizedSVD<monolithic,SVDType_> {
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
        FixedLambdaMonolithicSolver mono_solver(model,seed_);
        // select optimal smoothing level according to requested calibration strategy
        DVector<double> optimal_lambda;
        switch (calibration_) {
            case Calibration::off: {
                optimal_lambda = model.lambda();
            } break;
            case Calibration::gcv: {
                // select \lambda minimizing the GCV index
                ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                    mono_solver.init(lambda);
                    mono_solver.compute(X,rank);
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
                    mono_solver.init(lambda);
                    //fitting on the training set
                    mono_solver.compute(train_set.repeat(1,X.cols()).select(X),rank);
                    //evaluate the error on the test set
                    return mono_solver.reconstruction_error(test_set.repeat(1,X.cols()).select(X));
                };
                optimal_lambda =
                        calibration::KCV{n_folds_, seed_}.fit(model, lambda_grid_, cv_score);
            } break;
        }
        //Run using the optimal lambda
        mono_solver.init(optimal_lambda);
        //train on all the data
        mono_solver.compute(X,rank);

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
};


template<typename SVDType_>
class RegularizedSVD<monolithic,SVDType_,true>{
private:
    //algorithm parameters
    int seed_ = fdapde::random_seed;
    double tolerance_ = 1e-6;   // relative tolerance between Jnew and Jold, used as stopping criterion
    int max_iter_ = 100;

public:
    // constructors
    RegularizedSVD() = default;

    // solves \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top] retaining the first rank components
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {

        DMatrix<bool> W = !X.array().isNaN();
        DMatrix<double> U = DMatrix<double>::Zero(X.rows(),model.Psi().cols());

        FixedLambdaMonolithicSolver<ModelType,SVDType_> mono_solver(model,seed_);
        mono_solver.init(model.lambda()); //works only at fixed lambda

        for(int k = 1; k <= rank; ++k){
            //Majorization-Minimization scheme
            double Jold = std::numeric_limits<double>::max();
            double Jnew = 1.0;
            int j=0;
            while(!almost_equal(Jnew, Jold, tolerance_) && j < max_iter_){
                DMatrix<double> X_imputed = W.select(X,0)+(!W.array()).select(U*model.Psi().transpose(),0);
                X_imputed.rowwise() -= X_imputed.colwise().mean();
                //fit on the imputed data
                mono_solver.compute(X_imputed,k);
                U = mono_solver.scores()*mono_solver.loadings().transpose();
                //update
                j++;
                Jold = Jnew;
                Jnew = (W.select(X-U*model.Psi().transpose(),0)).squaredNorm()+(U*model.P()*U.transpose()).trace();
            }
        }
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
private:
    // let E*\Sigma*F^\top the reduced (rank r) SVD of X*\Psi*(D^{1})^\top, with D^{-1} the inverse of the cholesky
    // factor of \Psi^\top * \Psi + P(\lambda), then
    DMatrix<double> scores_;          // matrix E in the reduced SVD of X*\Psi*(D^{-1})^\top
    DMatrix<double> loadings_;        // \Sigma*F^\top*D^{-1} (PC functions expansion coefficients, L^2 normalized)
    DVector<double> loadings_norm_;   // L^2 norm of estimated fields
};

template<typename SVDType_>
class RegularizedSVD<sequential,SVDType_,true>{
private:
    // power iteration parameters
    double tolerance_ = 1e-6;   // relative tolerance between Jnew and Jold, used as stopping criterion
    int max_iter_ = 100;         // maximum number of allowed iterations
    int seed_ = fdapde::random_seed;
    //calibration
    int n_folds_ = 10;   // for a kcv calibration strategy, the number of folds
    DMatrix<double> lambda_grid_;

    template <typename ModelType>
    std::pair<DMatrix<double>,DMatrix<double>> MMscheme(const DMatrix<double>& X, ModelType& model, int rank,
                                                        PowerIteration<ModelType> &solver, DVector<double> lambda) {
        //init loadings and scores dimensions
        DMatrix<double> loadings(model.n_basis(), rank);
        DMatrix<double> scores(X.rows(), rank);
        DVector<double> loadings_norm(rank);
        //MM-scheme init
        DMatrix<bool> W = !X.array().isNaN();
        DMatrix<double> U = DMatrix<double>::Zero(X.rows(), model.n_basis());
        SVDType_ svd;
        for(int k = 1; k <= rank; ++k){
            //Majorization-Minimization scheme
            int j = 0;
            double Jold = std::numeric_limits<double>::max();
            double Jnew = 1;
            while (!almost_equal(Jnew, Jold, tolerance_) && j < max_iter_) {
                DMatrix<double> X_imputed = W.select(X, 0) + (!W.array()).select(U * model.Psi().transpose(), 0);
                X_imputed.rowwise() -= X_imputed.colwise().mean();
                //Sequential fPCA on the imputed data
                //->init with SVD
                if constexpr (is_rand_svd<SVDType_>{}) {
                    svd.compute(X_imputed, rank);
                } else {
                    svd.compute(X_imputed, Eigen::ComputeThinU | Eigen::ComputeThinV);
                }
                //->sequential estimation of the components
                for (int index = 0; index < k; index++) {
                    //fit on the imputed data
                    solver.compute(X_imputed, lambda, svd.matrixV().col(index));
                    //deflation
                    X_imputed -= solver.s() * solver.fn().transpose() * solver.f_norm();
                    //normalization
                    loadings.col(index) = solver.f();
                    scores.col(index) = solver.s() * solver.f_norm();
                    loadings_norm[index] = solver.f_norm();
                }
                U = scores.leftCols(k) * loadings.leftCols(k).transpose();
                //update
                j++;
                Jold = Jnew;
                Jnew = (W.select(X - U * model.Psi().transpose(), 0)).squaredNorm() +
                       (U * model.P(lambda) * U.transpose()).trace();
            }
        }
        return std::make_pair(scores, loadings);
    }

public:
    // constructors
    RegularizedSVD() = default;

    // solves \norm{X - U*\Psi^\top}_F^2 + Tr[U*P_{\lambda}(f)*U^\top] retaining the first rank components
    template <typename ModelType> void compute(const DMatrix<double>& X, ModelType& model, int rank) {
        //Init the solver
        PowerIteration<ModelType> solver(model,1e-6,20);
        solver.init(); //compute the factorization just once
        /*
        auto cv_score = [&](
                const DVector<double>& lambda,
                const core::BinaryVector<Dynamic>& train_set,
                const core::BinaryVector<Dynamic>& test_set) -> double {

            //fitting on the training set
            std::pair<DMatrix<double>,DMatrix<double>> solution = this->MMscheme<ModelType>(train_set.repeat(1,X.cols()).select(X), model, rank, solver, lambda);

            //evaluate the error on the test set
            DMatrix<double> S_m = model.Psi()*solution.second;
            S_m = S_m*S_m.transpose();
            return (test_set.repeat(1,X.cols()).select(X*(DMatrix<double>::Identity(X.cols(), X.cols()) - S_m))).squaredNorm() / (test_set.size() * X.cols());
        };
        */
        //final solution
        DVector<double> optimal_lambda = model.lambda(); //calibration::KCV{n_folds_, seed_}.fit(model, lambda_grid_, cv_score);
        std::pair<DMatrix<double>,DMatrix<double>> solution = MMscheme<ModelType>(X, model, rank, solver, optimal_lambda);
        // store results
        scores_ = solution.first;
        loadings_ = solution.second;
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
