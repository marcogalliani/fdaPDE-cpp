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

#ifndef __FDAPDE_NPRODE_H__
#define __FDAPDE_NPRODE_H__

#include "header_check.h"

namespace fdapde {

// Physics-informed smoothing with an ODE-residual (control) penalty: fits a d-dimensional
// trajectory to time-series observations while penalizing departures from a prior dynamics
// y' = f(t, y). Thin model wrapper around a time-stepping least-squares solver (ts_ls_ode),
// adding penalty-parameter selection via GCV, in the same spirit as SRPDE wraps its solvers.
template <typename VariationalSolver>
    requires(std::is_same_v<typename VariationalSolver::solver_category, ls_solver>)
class NPRODE {
   private:
    using solver_t = std::decay_t<VariationalSolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    static constexpr int n_lambda = solver_t::n_lambda;

   public:
    NPRODE() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    NPRODE(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) {
        discretize(penalty.get());
        analyze_data(formula, gf);
    }
    // modifiers
    template <typename... Args> void discretize(Args&&... args) { solver_.discretize(std::forward<Args>(args)...); }
    template <typename GeoFrame> void analyze_data(const std::string& formula, const GeoFrame& gf) {
        fdapde_assert(gf.n_layers() == 1);
        solver_.analyze_data(formula, gf);
    }
    void analyze_data(const vector_t& time_nodes, const matrix_t& y_obs) {
        solver_.analyze_data(time_nodes, y_obs);
    }
    // optional per-component state box constraints lb <= y(t) <= ub (honoured by the SQP fit policy)
    void set_state_bounds(const vector_t& lb, const vector_t& ub) { solver_.set_state_bounds(lb, ub); }
    void clear_state_bounds() { solver_.clear_state_bounds(); }
    bool has_state_bounds() const { return solver_.has_state_bounds(); }
    // fitting
    template <typename... Args> auto fit(Args&&... args) { return solver_.fit(std::forward<Args>(args)...); }

    // observers
    const vector_t& f() const { return solver_.f(); }                 // flattened trajectory
    vector_t fitted() const { return solver_.fitted(); }
    const vector_t& response() const { return solver_.response(); }
    const matrix_t& trajectory() const { return solver_.trajectory(); }   // m x d
    const matrix_t& control() const { return solver_.control(); }         // (m-1) x d defects
    double objective() const { return solver_.objective(); }
    bool converged() const { return solver_.converged(); }
    int n_iter() const { return solver_.n_iter(); }
    int n_obs() const { return solver_.n_obs(); }
    int n_components() const { return solver_.n_components(); }
    int n_nodes() const { return solver_.n_nodes(); }
    double rss() const { return solver_.rss(); }
    double edf(int r = 100, int seed = random_seed) { return solver_.edf(r, seed); }

    // Generalized Cross Validation index (penalty-parameter selection)
    struct gcv_t : public ScalarFieldBase<n_lambda, gcv_t> {
        using Base = ScalarFieldBase<n_lambda, gcv_t>;
        static constexpr int StaticInputSize = n_lambda;
        static constexpr int NestAsRef = 0;
        static constexpr int XprBits = 0;
        using Scalar = double;
        using InputType = Vector<Scalar, StaticInputSize>;
        using edf_cache_t = std::unordered_map<
          std::array<double, StaticInputSize>, double, internals::std_array_hash<double, StaticInputSize>>;

        gcv_t() noexcept = default;
        gcv_t(NPRODE* model, const edf_cache_t& edf_cache, int r, int seed) :
            model_(model), edf_cache_(edf_cache), r_(r), seed_(seed) { }
        gcv_t(NPRODE* model, const edf_cache_t& edf_cache) : gcv_t(model, edf_cache, 100, random_seed) { }
        gcv_t(NPRODE* model) : gcv_t(model, edf_cache_t()) { }
        gcv_t(NPRODE* model, int r, int seed) : gcv_t(model, edf_cache_t(), r, seed) { }

        template <typename InputType_>
            requires(internals::is_subscriptable<InputType_, int>)
        constexpr double operator()(const InputType_& lambda) {
            return internals::apply_index_pack<n_lambda>([&]<int... Ns_>() { return operator()(lambda[Ns_]...); });
        }
        template <typename... LambdaT>
            requires(std::is_convertible_v<LambdaT, double> && ...) && (sizeof...(LambdaT) == StaticInputSize)
        constexpr double operator()(LambdaT... lambda) {
            model_->fit(static_cast<double>(lambda)...);
            std::array<double, StaticInputSize> lambda_vec {static_cast<double>(lambda)...};
            if (edf_cache_.find(lambda_vec) == edf_cache_.end()) { edf_cache_[lambda_vec] = model_->edf(r_, seed_); }
            double n = model_->n_obs();
            double dor = n - edf_cache_.at(lambda_vec);   // residual degrees of freedom
            return (n / std::pow(dor, 2)) * model_->rss();
        }
        const edf_cache_t& edf_cache() const { return edf_cache_; }
        edf_cache_t& edf_cache() { return edf_cache_; }
       private:
        NPRODE* model_ = nullptr;
        edf_cache_t edf_cache_;
        int r_ = 100, seed_ = random_seed;
    };
    gcv_t gcv() { return gcv_t(this); }
    gcv_t gcv(const typename gcv_t::edf_cache_t& edf_cache) { return gcv_t(this, edf_cache); }
    gcv_t gcv(int r, int seed) { return gcv_t(this, r, seed); }
    gcv_t gcv(const typename gcv_t::edf_cache_t& edf_cache, int r, int seed) {
        return gcv_t(this, edf_cache, r, seed);
    }

   private:
    solver_t solver_;
};

}   // namespace fdapde

#endif   // __FDAPDE_NPRODE_H__
