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

#ifndef __FPIRLS_H__
#define __FPIRLS_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;

#include "../model_macros.h"
#include "../model_traits.h"
#include "distributions.h"
#include "regression_wrappers.h"
#include "srpde.h"
#include "strpde.h"

namespace fdapde {
namespace models {

// a general implementation of the Functional Penalized Iterative Reweighted Least Square (FPIRLS) algorithm
template <typename Model_> class FPIRLS {
   private:
    using Model = typename std::decay<Model_>::type;
    using SolverWrapper = RegressionModel<typename std::decay_t<Model_>::RegularizationType>;
    Model& m_;
    // algorithm's parameters
    double tolerance_;       // treshold on objective functional J to convergence
    std::size_t max_iter_;   // maximum number of iterations before forced stop
    std::size_t k_ = 0;      // FPIRLS iteration index
    SolverWrapper solver_;   // internal solver
   public:
    // constructor
    FPIRLS(Model& m, double tolerance, std::size_t max_iter) :
        m_(m), tolerance_(tolerance), max_iter_(max_iter) {
         if (!solver_) { // default solver initialization
            using SolverType = typename std::conditional<
              is_space_only<Model>::value, SRPDE,
              STRPDE<typename Model::RegularizationType, fdapde::monolithic> >::type;
            // define internal problem solver
            if constexpr (!is_space_time<Model_>::value)   // space-only
                solver_ = SolverType(m_.pde(), m_.sampling());
            else {   // space-time
                solver_ = SolverType(m_.pde(), m_.sampling(), m_.time_domain());
                if constexpr (is_space_time_parabolic<Model_>::value) { solver_.set_initial_condition(m_.s(), false); }
                if constexpr (is_space_time_separable<Model_>::value) {
                    solver_.set_temporal_locations(m_.time_locs());
                }
            }
            // solver initialization
            solver_.set_lambda(m_.lambda());
            solver_.set_spatial_locations(m_.locs());
            solver_.set_data(m_.data());   // possible covariates are passed from here	    
        }
    };

    // executes the FPIRLS algorithm
    void compute() {
	m_.fpirls_init();
        // objective functional value at consecutive iterations
        double J_old = tolerance_ + 1;
        double J_new = 0;
        while (k_ < max_iter_ && std::abs(J_new - J_old) > tolerance_) {
            m_.fpirls_compute_step();   // model specific computation of py_ and pW_
            // solve weighted least square problem
            // \argmin_{\beta, f} [ \norm(W^{1/2}(y - X\beta - f_n))^2 + \lambda \int_D (Lf - u)^2 ]
            solver_.data().template insert<double>(OBSERVATIONS_BLK, m_.py());
            solver_.data().template insert<double>(WEIGHTS_BLK, m_.pW());
	    // update solver and solve
	    solver_.init();
            solver_.solve();
            m_.fpirls_update_step(solver_.fitted(), solver_.beta());   // model specific update step
            // update value of the objective functional J = data_loss + \int_D (Lf-u)^2
            double J = m_.data_loss() + m_.lambda_D()*solver_.g().dot(m_.R0() * solver_.g());
            // prepare for next iteration
            k_++; J_old = J_new; J_new = J;
        }
        return;
    }
    // sets an externally defined solver
    template <typename T> void set_solver(T&& solver) { solver_ = solver; }
    // getters
    std::size_t n_iter() const { return k_; }
    const SolverWrapper& solver() const { return solver_; }
};

}   // namespace models
}   // namespace fdapde

#endif   // __FPIRLS_H__