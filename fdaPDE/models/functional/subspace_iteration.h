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

#ifndef __SUBSPACE_ITERATION_H__
#define __SUBSPACE_ITERATION_H__

#include <fdaPDE/utils.h>
#include "../model_traits.h"
#include "../regression/gcv.h"
#include "../regression/srpde.h"
#include "../regression/stochastic_edf.h"
#include "../regression/strpde.h"
using fdapde::models::GCV;

namespace fdapde {
namespace models {

template <typename Model_> class SubspaceIteration {
private:
    using Model = typename std::decay<Model_>::type;
    using SolverType = typename std::conditional<
            is_space_only<Model>::value, SRPDE, STRPDE<typename Model::RegularizationType, fdapde::monolithic>>::type;
    static constexpr int n_lambda = Model::n_lambda;
    GCV gcv_;   // GCV functor
    std::vector<double> gcv_scores_;
    // algorithm's parameters
    double tolerance_ = 1e-5;   // treshold on |Jnew - Jold| used as stopping criterion
    int max_iter_ = 50;         // maximum number of iterations before forced stop
    int k_ = 0;                 // iteration index
    std::vector<SolverType> solvers_;         // smooth each PC separately
    int seed_ = fdapde::random_seed;
    int n_mc_samples_ = 100;
    int rank_;

    DMatrix<double> S_;    // estimated score matrix
    DMatrix<double> Fn_;   // field evaluation at data location \frac{\Psi*f}{\norm{f}_{L^2}}
    DMatrix<double> F_;    // field basis expansion at convergence
    DVector<double> f_norm_;        // L^2 norm of estimated field at converegence
public:
    // constructors
    SubspaceIteration() = default;
    SubspaceIteration(const Model& m, double tolerance, int max_iter, int seed) :
            tolerance_(tolerance), max_iter_(max_iter),
            seed_((seed == fdapde::random_seed) ? std::random_device()() : seed) {
        // initialize internal smoothing solver
        solvers_.resize(1);
        if constexpr (is_space_only<SolverType>::value) {
            solvers_[0] = SolverType(m.pde(), m.sampling()); }
        else {
            solvers_[0] = SolverType(m.pde(), m.time_pde(), m.sampling());
            solvers_[0].set_temporal_locations(m.time_locs());
        }
        solvers_[0].set_spatial_locations(m.locs());
        return;
    };
    template <typename ModelType>
    SubspaceIteration(const ModelType& m, double tolerance, int max_iter) :
            SubspaceIteration(m, tolerance, max_iter, fdapde::random_seed) { }
    // executes the power iteration algorithm on data X and smoothing parameter \lambda, starting from F0
    void compute(const DMatrix<double>& X, int rank,const DMatrix<double>& lambdas, const DMatrix<double>& F0) {
        //rows of lambda contains different PCs, cols: space & time smoothing
        rank_ = rank;
        auto tmp = solvers_.front();
        solvers_.assign(rank_, tmp);
        for (int i = 0; i < rank_; ++i) {
            solvers_[i].set_lambda(SVector<n_lambda>(lambdas.row(i)));
            solvers_[i].init();
        }
        gcv_ = solvers_.back().template gcv<StochasticEDF>(n_mc_samples_, seed_);
        // initialization
        Fn_.resize(X.cols(),rank_);
        F_.resize(solvers_[0].n_basis(),rank_);
        S_.resize(X.rows(),rank_);
        k_ = 0;   // reset iteration counter
        double Jold = std::numeric_limits<double>::max();
        double Jnew = 1;
        Fn_ = F0;   // set starting point
        while (!fdapde::almost_equal(Jnew, Jold, tolerance_) && k_ < max_iter_) {
            // computation of the scores
            S_ = X * Fn_;
            // normalization
            Eigen::HouseholderQR<DMatrix<double>> qr(S_);
            S_ = qr.householderQ()*DMatrix<double>::Identity(S_.rows(),rank_);
            // compute loadings (solve smoothing problem)
            double roughness_penalty = 0;
            for (int i = 0; i < rank_; ++i){
                solvers_[i].data().template insert<double>(OBSERVATIONS_BLK, X.transpose() * S_.col(i));   // X^\top*s
                solvers_[i].solve();
                F_.col(i) = solvers_[i].f();
                Fn_.col(i) = solvers_[i].fitted();
                roughness_penalty += solvers_[i].ftPf(lambdas.row(i));
            }
            // prepare for next iteration
            k_++;
            // update value of discretized functional: \norm{X - s*f_n^\top}_F + f^\top*P(\lambda)*f
            Jold = Jnew;
            Jnew = (X - S_ * Fn_.transpose()).squaredNorm() + roughness_penalty;
        }
        // store results
        f_norm_ = (F_.transpose()*solvers_.front().R0()*F_).diagonal().cwiseSqrt(); // L^2 norm of estimated field
        F_ = F_.array().rowwise() / f_norm_.transpose().array(); // estimated field (L^2 normalized)
        Fn_ = solvers_.front().Psi() * F_;   // evaluation of (L^2 unitary norm) estimated field at data locations
        S_ = S_.array().rowwise() * f_norm_.transpose().array();
        return;
    }
    //calibration
    double gcv(){
        return gcv_.eval();
    }
    // getters
    const DMatrix<double>& F() const { return F_; }   // loadings matrix
    const DMatrix<double>& S() const { return S_; }   // scores vector
    const DMatrix<double>& Fn() const { return Fn_; }
    int n_iter() const { return k_; }
    DVector<double> f_norm() const { return f_norm_; }
    inline DVector<double> f_squaredNorm() const { return f_norm_.array().pow(2); }
    // setters
    void set_tolerance(double tolerance) { tolerance_ = tolerance; }
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
    void set_seed(int seed) { seed_ = seed; }
};




template <typename Model_> class FixedLambdaSubspaceIteration {
private:
    using Model = typename std::decay<Model_>::type;
    using SolverType = typename std::conditional<
            is_space_only<Model>::value, SRPDE, STRPDE<typename Model::RegularizationType, fdapde::monolithic>>::type;
    static constexpr int n_lambda = Model::n_lambda;
    GCV gcv_;   // GCV functor
    std::vector<double> gcv_scores_;
    // algorithm's parameters
    double tolerance_ = 1e-5;   // treshold on |Jnew - Jold| used as stopping criterion
    int max_iter_ = 50;         // maximum number of iterations before forced stop
    int k_ = 0;                 // iteration index
    SolverType solver_;         // smooth each PC separately
    int seed_ = fdapde::random_seed;
    int n_mc_samples_ = 100;
    int rank_;

    DMatrix<double> S_;    // estimated score matrix
    DMatrix<double> Fn_;   // field evaluation at data location \frac{\Psi*f}{\norm{f}_{L^2}}
    DMatrix<double> F_;    // field basis expansion at convergence
    DVector<double> f_norm_;        // L^2 norm of estimated field at converegence
public:
    // constructors
    FixedLambdaSubspaceIteration() = default;
    FixedLambdaSubspaceIteration(const Model& m, double tolerance, int max_iter, int seed) :
            tolerance_(tolerance), max_iter_(max_iter),
            seed_((seed == fdapde::random_seed) ? std::random_device()() : seed) {
        // initialize internal smoothing solver
        if constexpr (is_space_only<SolverType>::value) {
            solver_ = SolverType(m.pde(), m.sampling()); }
        else {
            solver_ = SolverType(m.pde(), m.time_pde(), m.sampling());
            solver_.set_temporal_locations(m.time_locs());
        }
        solver_.set_spatial_locations(m.locs());
        return;
    };
    template <typename ModelType>
    FixedLambdaSubspaceIteration(const ModelType& m, double tolerance, int max_iter) :
            FixedLambdaSubspaceIteration(m, tolerance, max_iter, fdapde::random_seed) { }
    // executes the power iteration algorithm on data X and smoothing parameter \lambda, starting from F0
    void compute(const DMatrix<double>& X, int rank,const DVector<double>& lambda, const DMatrix<double>& F0) {
        //rows of lambda contains different PCs, cols: space & time smoothing
        rank_ = rank;
        gcv_ = solver_.template gcv<StochasticEDF>(n_mc_samples_, seed_);
        //initialise the smoothing solver
        solver_.set_lambda(lambda);
        solver_.init();
        // initialization
        Fn_.resize(X.cols(),rank_);
        F_.resize(solver_.n_basis(),rank_);
        S_.resize(X.rows(),rank_);
        k_ = 0;   // reset iteration counter
        double Jold = std::numeric_limits<double>::max();
        double Jnew = 1;
        Fn_ = F0;   // set starting point
        while (!fdapde::almost_equal(Jnew, Jold, tolerance_) && k_ < max_iter_) {
            // computation of the scores
            S_ = X * Fn_;
            // normalization
            Eigen::JacobiSVD<DMatrix<double>> svd(S_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            S_ = svd.matrixU();
            // compute loadings (solve smoothing problem)
            double roughness_penalty = 0;
            for (int i = 0; i < rank_; ++i){
                solver_.data().template insert<double>(OBSERVATIONS_BLK, X.transpose() * S_.col(i));   // X^\top*s
                solver_.solve();
                F_.col(i) = solver_.f();
                Fn_.col(i) = solver_.fitted();
                roughness_penalty += solver_.ftPf(lambda);
            }
            // prepare for next iteration
            k_++;
            // update value of discretized functional: \norm{X - s*f_n^\top}_F + f^\top*P(\lambda)*f
            Jold = Jnew;
            Jnew = (X - S_ * Fn_.transpose()).squaredNorm() + roughness_penalty;
        }
        // store results
        f_norm_ = (F_.transpose()*solver_.R0()*F_).diagonal().cwiseSqrt(); // L^2 norm of estimated field
        F_ = F_.array().rowwise() / f_norm_.transpose().array(); // estimated field (L^2 normalized)
        Fn_ = solver_.Psi() * F_;   // evaluation of (L^2 unitary norm) estimated field at data locations
        S_ = S_.array().rowwise() * f_norm_.transpose().array();
        return;
    }
    //calibration
    double gcv(const DMatrix<double>& X){
        DMatrix<double> S_norm = S_.array().rowwise() / f_norm_.transpose().array();
        DMatrix<double> F_unnorm = F_.array().rowwise() * f_norm_.transpose().array();
        int n_locs = X.cols();
        return n_locs*(S_norm.transpose()*X-(solver_.Psi() * F_unnorm).transpose()).squaredNorm()/std::pow(n_locs-gcv_.eval_edfs(),2);
    }
    // getters
    const DMatrix<double>& F() const { return F_; }   // loadings matrix
    const DMatrix<double>& S() const { return S_; }   // scores vector
    const DMatrix<double>& Fn() const { return Fn_; }
    int n_iter() const { return k_; }
    DVector<double> f_norm() const { return f_norm_; }
    inline DVector<double> f_squaredNorm() const { return f_norm_.array().pow(2); }
    // setters
    void set_tolerance(double tolerance) { tolerance_ = tolerance; }
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
    void set_seed(int seed) { seed_ = seed; }
};




}   // namespace models
}   // namespace fdapde

#endif //__SUBSPACE_ITERATION_H__
