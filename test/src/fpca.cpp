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

#include <array>
#include <cmath>
#include <random>
#include <vector>

using namespace fdapde;

using vector_t = Eigen::Matrix<double, Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

namespace {

// a Laplacian eigenfunction cos(a x) cos(b y) sampled at the mesh nodes
vector_t eigenfunction(const matrix_t& nodes, double a, double b) {
    vector_t v(nodes.rows());
    for (int i = 0; i < nodes.rows(); ++i) { v[i] = std::cos(a * nodes(i, 0)) * std::cos(b * nodes(i, 1)); }
    return v;
}

// synthetic functional dataset: a rank-3 signal built from three Laplacian eigenfunctions of
// decreasing score variance (so the third component is the weakest) plus i.i.d. Gaussian noise.
// Returns the data matrix Y (n_nodes x n_units) and fills `signal` with the noise-free signal.
matrix_t make_fpca_data(const Triangulation<2, 2>& D, int n_units, double noise_frac, matrix_t& signal) {
    const matrix_t& nodes = D.nodes();
    const double pi = M_PI;
    matrix_t F(nodes.rows(), 3);
    F.col(0) = eigenfunction(nodes, 1 * pi, 1 * pi);
    F.col(1) = eigenfunction(nodes, 1 * pi, 3 * pi);
    F.col(2) = eigenfunction(nodes, 4 * pi, 2 * pi);
    double dr = F.maxCoeff() - F.minCoeff();
    std::mt19937 rng(4513);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::array<double, 3> sd = {0.4 * dr, 0.3 * dr, 0.2 * dr};
    matrix_t scores(n_units, 3);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < n_units; ++i) { scores(i, j) = sd[j] * nd(rng); }
    }
    signal = F * scores.transpose();   // n_nodes x n_units
    matrix_t Y = signal;
    std::normal_distribution<double> ne(0.0, noise_frac * dr);
    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = 0; j < Y.cols(); ++j) { Y(i, j) += ne(rng); }
    }
    return Y;
}

// a logarithmically spaced grid 10^lo_exp ... 10^hi_exp
std::vector<double> log_grid(double lo_exp, double hi_exp, double step) {
    std::vector<double> g;
    for (double e = lo_exp; e <= hi_exp + 1e-9; e += step) { g.push_back(std::pow(10.0, e)); }
    return g;
}

// extracted results of a Laplacian fPCA fit (copies so the model can be destroyed)
struct fpca_fit {
    matrix_t lambda;   // selected smoothing level per component
    matrix_t S;        // scores       (n_units x rank)
    matrix_t F;        // loadings      (n_dofs  x rank)
    matrix_t Fn;       // loadings at locations (n_locs x rank)
};

// builds a simple-Laplacian-penalized fPCA over `D`, fits `rank` components on the GCV grid with the
// given solver policy, and returns the extracted quantities.
template <typename Policy>
fpca_fit run_fpca(Triangulation<2, 2>& D, const matrix_t& Y, int rank,
                  const std::vector<double>& grid, Policy policy) {
    // simple laplacian penalty
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto L = integral(D)(u * v);
    // data: multi-column functional response loaded as a block column
    GeoFrame data(D);
    auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", Y);
    // model
    fPCA<internals::fe_ls_elliptic> model;
    model.discretize(fe_ls_elliptic{a, L}.get());
    model.analyze_data("y", data);
    // exact SVD initial guess keeps the run deterministic; GCV calibration is auto-selected
    model.fit(rank, grid, ComputeXactSVD, policy);
    return fpca_fit{model.lambda(), model.S(), model.F(), model.Fn()};
}

// flattens the (rank x 1) lambda matrix to a vector of selected smoothing levels
std::vector<double> selected_lambdas(const matrix_t& lambda) {
    std::vector<double> out;
    for (int i = 0; i < lambda.size(); ++i) { out.push_back(lambda(i)); }
    return out;
}

}   // namespace

// fitted quantities have the expected shapes (sequential power-iteration solver)
TEST(fpca, shapes) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    const int n_units = 50, rank = 3;
    matrix_t signal;
    matrix_t Y = make_fpca_data(D, n_units, 0.1, signal);

    auto fit = run_fpca(D, Y, rank, log_grid(-4, 2, 0.5), fpca_power_solver {});

    EXPECT_EQ(fit.S.rows(), n_units);
    EXPECT_EQ(fit.S.cols(), rank);
    EXPECT_EQ(fit.F.cols(), rank);
    EXPECT_EQ(fit.Fn.rows(), D.n_nodes());
    EXPECT_EQ(static_cast<int>(selected_lambdas(fit.lambda).size()), rank);
}

// regression: the sequential GCV must not collapse to the largest (most over-smoothing) lambda on
// the weaker later components. Before the full-reconstruction + score-dof fix the second and third
// components latched onto the maximum of the grid.
TEST(fpca, sequential_gcv_no_high_lambda_collapse) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;
    matrix_t Y = make_fpca_data(D, 50, 0.1, signal);
    std::vector<double> grid = log_grid(-4, 2, 0.5);   // extends well into the over-smoothing region

    auto fit = run_fpca(D, Y, 3, grid, fpca_power_solver {});

    double lambda_max = grid.back();
    for (double l : selected_lambdas(fit.lambda)) { EXPECT_LT(l, lambda_max); }
}

// regression: the subspace solver selects an independent lambda per component (cyclic coordinate
// descent on the summed, energy-guarded GCV) and likewise keeps every component off the
// over-smoothing end of the grid.
TEST(fpca, subspace_gcv_per_component_no_collapse) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;
    matrix_t Y = make_fpca_data(D, 50, 0.1, signal);
    std::vector<double> grid = log_grid(-4, 2, 0.5);

    auto fit = run_fpca(D, Y, 3, grid, fpca_subspace_experimental_solver {});

    std::vector<double> lams = selected_lambdas(fit.lambda);
    EXPECT_EQ(static_cast<int>(lams.size()), 3);
    double lambda_max = grid.back();
    for (double l : lams) { EXPECT_LT(l, lambda_max); }
}

// the subspace solver enforces orthogonal scores (S^T S diagonal): the off-diagonal Gram entries
// are negligible relative to the diagonal. This is the structural guarantee that distinguishes the
// subspace estimator from the sequential one.
TEST(fpca, subspace_scores_orthogonal) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    const int rank = 3;
    matrix_t signal;
    matrix_t Y = make_fpca_data(D, 50, 0.1, signal);

    auto fit = run_fpca(D, Y, rank, log_grid(-4, 2, 0.5), fpca_subspace_experimental_solver {});

    matrix_t G = fit.S.transpose() * fit.S;   // rank x rank Gram matrix of the scores
    for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
            if (i == j) { continue; }
            double normalized_off = std::abs(G(i, j)) / std::sqrt(G(i, i) * G(j, j));
            EXPECT_LT(normalized_off, 1e-4);
        }
    }
}

// end-to-end sanity: the rank-3 reconstruction recovers the noise-free signal subspace, i.e. it
// explains the large majority of the signal energy, for both solvers.
TEST(fpca, reconstruction_recovers_signal) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;   // n_nodes x n_units
    matrix_t Y = make_fpca_data(D, 50, 0.1, signal);
    std::vector<double> grid = log_grid(-4, 2, 0.5);

    for (bool subspace : {false, true}) {
        fpca_fit fit = subspace ? run_fpca(D, Y, 3, grid, fpca_subspace_experimental_solver {})
                                : run_fpca(D, Y, 3, grid, fpca_power_solver {});
        matrix_t Yhat = fit.Fn * fit.S.transpose();   // n_nodes x n_units reconstruction
        double rel = (signal - Yhat).squaredNorm() / signal.squaredNorm();
        EXPECT_LT(rel, 0.5);
    }
}