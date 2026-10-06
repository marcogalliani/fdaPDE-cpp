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

#ifndef __FSVT_H__
#define __FSVT_H__

#include <tuple>

#include "header_check.h"

namespace fdapde {
namespace internals {

// implementing functional singular value thresholding
template <typename fPCASolver> class fsvt_impl {
   private:
    using fpca_t = std::decay_t<fPCASolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
   public:
    using smoother_t = typename fPCASolver::smoother_t;
    static constexpr int n_lambda = smoother_t::n_lambda;

    fsvt_impl() noexcept = default;
    fsvt_impl(fPCASolver& fpca) noexcept :
        fpca_(std::addressof(fpca)), smoother_(*fpca.smoother()), n_dofs_(fpca.smoother()->n_dofs()){ }

    template <typename DataT>
    auto fit(const DataT& data, double sing_val_th, int max_rank, const std::vector<double>& lambda_grid, int flag) {
        matrix_t X = data.transpose();         // create temporary of mapped data
        binary_t nan_pattern = na_matrix(X);   // compute missingness pattern
        n_locs_ = X.cols(), n_units_ = X.rows(); singular_val_threshold_ = sing_val_th; max_rank_ = max_rank;
        int calibration = (flag & 0b00110);   // detect calibration strategy

        matrix_t Xn0 = matrix_t::Zero(n_units_, n_dofs_);
        std::vector<double> opt_lambda(n_lambda);
        switch (calibration) {
        case 0: {   // no calibration
            fdapde_assert(lambda_grid.size() == n_lambda);
            std::copy(lambda_grid.begin(), lambda_grid.end(), opt_lambda.begin());
        } break;
        default: {
            throw std::runtime_error("Unrecognized calibration option.");
        }
        } //switch
        // fit with optimal lambda
        const auto& [U, Sigma, V] = solve_(X, nan_pattern, Xn0, opt_lambda, flag);
        //store results
        sing_val_.resize(Sigma.size());
        V_.resize(n_dofs_, Sigma.size());
        U_.resize(n_units_, Sigma.size());
        lambda_.resize(n_lambda);
        for (int j = 0; j < n_lambda; ++j) { lambda_(j) = opt_lambda[j]; }
        U_ = U; sing_val_= Sigma; V_= V;
        return std::make_tuple(U_, sing_val_, V_);
    }
    // observers
    const vector_t& singularValues() const { return sing_val_; }
    const matrix_t& U() const { return U_; }
    const matrix_t& V() const { return V_; }
    matrix_t Vn() const { return smoother_.Psi() * V_; }
    const vector_t& lambda() const { return lambda_; }
    int rank() const { return sing_val_.size(); }
   private:
    // the solve_ method implements the MM loop
    template <typename LambdaT>
        requires(internals::is_subscriptable<LambdaT, int>)
    auto solve_(const matrix_t& X, const binary_t& nan, const matrix_t& Xh0, const LambdaT lambda, int flag) {
        // init results
        matrix_t U, V;
        vector_t Sigma;
        // init algorithm
        matrix_t Xh = Xh0;
        matrix_t Xhn = Xh * smoother_.Psi().transpose();
        // smoothing matrix
        // normalise results with respect to the norm induced by the smoother: \norm{f} = \sqrt{f^\top*(\Psi^\top\Psi + P_{\lambda})*f}
        vector_t lumped_mass = lump(smoother_.mass()).diagonal(); // use mass lumping for efficiency
        // WARNING: only supports single lambda (e.g., fe_ls_separable not supported)
        // possible solution: implemente lumped mass within each solver
        Eigen::SparseMatrix<double> smooth_mat = smoother_.Psi().transpose() * smoother_.Psi() + lambda[0]*smoother_.stiff()*lumped_mass.cwiseInverse().asDiagonal()*smoother_.stiff();
        // MM scheme
        /* Majorization: -> impute the data with previous estimate
         * Minimisation: -> proximal operator (singular value thresholding)
         */
        // iterate until the MM scheme converges
        int n_iter = 0;
        double proj_err = tol_+1;
        while (proj_err > tol_ && n_iter < max_iter_) {
            // imputation update
            matrix_t Xn = (~nan).select(X, Xhn);
            // singular value thresholding
            // fPCA on imputed data (using a large enough rank)
            auto [v, u] = fpca_->fit(Xn.transpose(), max_rank_, lambda, flag & 0x1); //always run with no calibration
            // assemble singular value decomposition
            vector_t sigma(max_rank_);
            for (int i = 0; i < max_rank_; ++i) {
                // normalisation
                sigma(i) = std::sqrt(v.col(i).dot(smooth_mat * v.col(i))); // norm w.r.t. the modified inner product
                v.col(i) /= sigma(i);
                sigma(i) *= u.col(i).norm(); // norm l2
                u.col(i) /= u.col(i).norm();
            }
            // shrink sigma values
            sigma = (sigma.array() - singular_val_threshold_).max(0.0).matrix();
            // find rank
            int rank_kept = 0;
            for (rank_kept = 0; rank_kept < sigma.size(); ++rank_kept) {
                // Use a small tolerance for floating point comparison
                if (sigma(rank_kept) < 1e-15) {
                    break; // Stop at the first value that is effectively zero
                }
            }
            // Reconstruct Xh using the shrunken singular values
            if (rank_kept > 0) {
                Xh = u.leftCols(rank_kept) * sigma.head(rank_kept).asDiagonal() * v.leftCols(rank_kept).transpose();
            } else {
                // If rank is 0, the result is a zero matrix
                Xh = matrix_t::Zero(n_units_, n_dofs_);
            }
            // prepare for next iteration
            n_iter++;
            Xhn = Xh * smoother_.Psi().transpose();
            proj_err = (~nan).select(X - Xhn, 0).norm() / Xhn.norm();
            // save results
            U = u.leftCols(rank_kept);
            V = v.leftCols(rank_kept);
            Sigma = sigma.head(rank_kept);
        }
        return std::make_tuple(U, Sigma, V);
    }
    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    int n_folds_ = 5;
    int n_mc_samples_ = 100;       // to estimate the trace of the smoothing matrix
    fpca_t* fpca_;

    // algorithm parameters
    int max_rank_ = 100;                 // maximum rank possible
    double singular_val_threshold_ = 0.1;

    smoother_t smoother_;
    matrix_t U_, V_;
    vector_t sing_val_;
    vector_t lambda_;              // selected PCs smoothing level
    matrix_t gcv_scores_;          // gcv scores (#lambda_grid-by-rank matrix)

    // MM scheme parameters
    double tol_ = 1e-4;
    int max_iter_ = 100;
};

}   // namespace internals

  
template <typename VariationalSolver> class fSVT {
   private:
    using smoother_t = std::decay_t<VariationalSolver>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using data_t   = Eigen::Map<const Eigen::Matrix<double, Dynamic, Dynamic, Eigen::ColMajor>>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    static constexpr int n_lambda = smoother_t::n_lambda;
   public:
    fSVT() noexcept = default;
    template <typename GeoFrame, typename Penalty>
    fSVT(const std::string& colname, const GeoFrame& gf, Penalty&& penalty) noexcept : smoother_(), data_() {
        discretize(penalty.get());
        analyze_data(colname, gf);
    }
    template <typename... Args> void discretize(Args&&... args) {
        smoother_.discretize(std::forward<Args>(args)...);
        n_dofs_ = smoother_.n_dofs();
	return;
    }
    template <typename GeoFrame> void analyze_data(const std::string& colname, const GeoFrame& gf) {
        fdapde_assert(gf.n_layers() == 1);
        data_ = gf[0].data().template col<double>(colname).as_matrix();
        n_locs_ = data_.rows();
        n_units_ = data_.cols();
	    smoother_.analyze_data(gf, vector_t::Ones(gf[0].rows()).asDiagonal());
        // detect if data_ has at least one missing value
        has_nan_ = false;
        for (int i = 0; i < n_locs_; ++i) {
            for (int j = 0; j < n_units_; ++j) {
                if (std::isnan(data_(i, j))) {
                    has_nan_ = true;
                    break;
                }
            }
        }
	return;
    }

    template <typename LambdaT, typename Policy = fpca_subspace_solver>
        requires(internals::is_vector_like_v<LambdaT>)
    auto fit(double sing_val_th, int max_rank, const LambdaT& lambda_grid, int flag = ComputeRandSVD, Policy policy = Policy()) {
        fdapde_assert(lambda_grid.size() % n_lambda == 0);
        auto solver_ = policy.get(smoother_);   // instantiate solver implementation
        // initialise the model
        internals::fsvt_impl mm_scheme(solver_);
        // fit
        const auto& [U, Sigma, V] = mm_scheme.fit(data_, sing_val_th, max_rank, lambda_grid, flag);
        V_.resize(n_dofs_, Sigma.size());
        U_.resize(n_units_, Sigma.size());
        U_ = std::move(U);
        V_ = std::move(V);
        sing_val_ = std::move(Sigma);
        lambda_ = mm_scheme.lambda();

        return std::make_tuple(U_, sing_val_, V_);
    }
    // observers
    const vector_t& singularValues() const { return sing_val_; }
    const matrix_t& U() const { return U_; }
    const matrix_t& V() const { return V_; }
    matrix_t Vn() const { return smoother_.Psi() * V_; }
    const vector_t& lambda() const { return lambda_; }
    int rank() const { return sing_val_.size(); }
   private:
    matrix_t data_;         // mapped geoframe data
    smoother_t smoother_;   // variational solver used in the smoothing step
    bool has_nan_;

    int n_locs_ = 0, n_units_ = 0, n_dofs_ = 0;
    matrix_t U_, V_;
    vector_t sing_val_;
    vector_t lambda_;              // selected smoothing level
    matrix_t gcv_scores_;          // gcv scores (#lambda_grid-by-rank matrix)
};

// deduction guide
template <typename GeoFrame, typename Penalty>
fSVT(const std::string& colname, const GeoFrame& gf, Penalty&& solver) -> fSVT<typename Penalty::solver_t>;

}   // namespace fdapde

#endif   // __FSVT_H__
