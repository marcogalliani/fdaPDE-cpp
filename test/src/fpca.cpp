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
using fdapde::test::almost_equal;

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
    auto& layer = data.template insert_scalar_layer<POINT>("layer", MESH_NODES);
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

// test 1
//    mesh:         unit_square_60
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    solver:       subspace
TEST(fpca, test_01) {
    // geometry
    std::string mesh_path = "../data/mesh/unit_square_40/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    // load the data matrix (assume that X is an (n_units,n_locs) data matrix)
    std::string data_path = "../data/fpca/01/";
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> X = read_csv<double>(data_path + "y.csv").as_matrix();
    l1.load_blk("X", X.transpose());
    // physics
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    fPCA m("X", data, fe_ls_elliptic(a, F));

    // fit
    std::vector<double> lambda_grid(5);
    for (int i = 0; i < 5; ++i) { lambda_grid[i] = std::pow(10, -4.0 +  i);}
    m.fit(
        /* n_comp = */ 3,
        lambda_grid,
        /* options = */ ComputeRandSVD | OptimizeGCV,
        fpca_subspace_solver()
        );
    EXPECT_TRUE(almost_equal<double>(m.F().col(0), data_path + "f1.mtx") || almost_equal<double>(-m.F().col(0), data_path + "f1.mtx"));
    EXPECT_TRUE(almost_equal<double>(m.F().col(1), data_path + "f2.mtx") || almost_equal<double>(-m.F().col(1), data_path + "f2.mtx"));
    EXPECT_TRUE(almost_equal<double>(m.F().col(2), data_path + "f3.mtx") || almost_equal<double>(-m.F().col(2), data_path + "f3.mtx"));
    EXPECT_EQ(m.objective_history().size(), 3);
    EXPECT_EQ(m.iterations().size(), 3);
    EXPECT_EQ(m.monotone().size(), 3);
}

// check vector-valued GCV grids for every fPCA solver policy
TEST(fpca, vector_grid_all_solvers) {
    auto D = Triangulation<2, 2>::Rectangle(0, 1, 0, 1, 4, 4);
    GeoFrame data(D);
    auto& layer = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    Eigen::MatrixXd X(D.n_nodes(), 8);
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) { X(i, j) = std::sin(0.3 * i + 0.5 * j) + std::cos(0.2 * i - 0.7 * j); }
    }
    X = (X.colwise() - X.rowwise().mean()).eval();
    layer.load_blk("X", X);
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    const std::vector<double> grid {0.01, 0.1};
    auto check_policy = [&](auto policy) {
        fPCA model("X", data, fe_ls_elliptic(a, F));
        model.fit(2, grid, ComputeXactSVD | OptimizeGCV, policy);
        // every loading must be finite after searching the vector grid
        EXPECT_TRUE(model.F().array().isFinite().all());
        // every score must be finite after fitting the selected penalties
        EXPECT_TRUE(model.S().array().isFinite().all());
        for (int i = 0; i < model.lambda().size(); ++i) {
            // the selected penalty for each component must belong to the supplied grid
            EXPECT_TRUE(std::find(grid.begin(), grid.end(), model.lambda().data()[i]) != grid.end());
        }
    };
    check_policy(fpca_power_solver());
    check_policy(fpca_subspace_solver());
    check_policy(fpca_direct_solver());
}

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

// ---------------------------------------------------------------------------------------------------------------------
// partially observed data
// ---------------------------------------------------------------------------------------------------------------------

namespace {

// sets a fraction of the entries of Y to NaN, uniformly at random
matrix_t remove_entries(const matrix_t& Y, double frac, unsigned seed) {
    matrix_t Yna = Y;
    std::mt19937 rng(seed);
    std::bernoulli_distribution missing(frac);
    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = 0; j < Y.cols(); ++j) {
            if (missing(rng)) { Yna(i, j) = std::numeric_limits<double>::quiet_NaN(); }
        }
    }
    return Yna;
}

// builds a simple-Laplacian-penalized fPCA over `D` on the (n_nodes x n_units) data Y and calls f(model)
template <typename F> void with_fpca(Triangulation<2, 2>& D, const matrix_t& Y, F&& f) {
    FeSpace Vh(D, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(u), grad(v)));
    ZeroField<2> zero;
    auto L = integral(D)(zero * v);
    GeoFrame data(D);
    auto& layer = data.template insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", Y);
    fPCA<internals::fe_ls_elliptic> model;
    model.discretize(fe_ls_elliptic{a, L}.get());
    model.analyze_data("y", data);
    f(model);
}

// relative error of the reconstruction mean + Fn * S^T with respect to the noise-free signal
template <typename Model> double reconstruction_error(const Model& model, const matrix_t& signal) {
    matrix_t Yhat = model.Fn() * model.S().transpose();
    Yhat.colwise() += model.center_locs();
    return (signal - Yhat).squaredNorm() / signal.squaredNorm();
}

}   // namespace

// MM scheme at a fixed smoothing level: 20% missing entries, the rank-3 fit recovers the signal
TEST(fpca, missing_data_fixed_lambda) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    const int rank = 3;
    matrix_t signal;
    matrix_t Y = remove_entries(make_fpca_data(D, 50, 0.1, signal), 0.2, 2711);
    with_fpca(D, Y, [&](auto& model) {
        model.fit(rank, std::vector<double> {1e-2}, ComputeXactSVD);
        EXPECT_EQ(model.F().cols(), rank);
        EXPECT_EQ(model.S().rows(), 50);
        EXPECT_EQ(model.lambda().rows(), rank + 1);   // row 0 is the smoothing level of the mean
        EXPECT_TRUE(model.F().array().isFinite().all() && model.S().array().isFinite().all());
        EXPECT_TRUE(model.converged());
        EXPECT_EQ(static_cast<int>(model.n_iter().size()), rank);
        EXPECT_LT(reconstruction_error(model, signal), 0.1);
    });
}

// the loadings returned for missing data are L^2-normalised and loadings_norm() refers to them
TEST(fpca, missing_data_loadings_normalised) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;
    matrix_t Y = remove_entries(make_fpca_data(D, 50, 0.1, signal), 0.2, 2711);
    FeSpace Vh(D, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);
    auto M = integral(D)(u * v).assemble();   // mass matrix
    with_fpca(D, Y, [&](auto& model) {
        model.fit(2, std::vector<double> {1e-2}, ComputeXactSVD);
        for (int i = 0; i < 2; ++i) {
            EXPECT_NEAR(model.F().col(i).dot(M * model.F().col(i)), 1.0, 1e-8);
            // scores * norm = unnormalised scores: the stored norm is the one of the final fit
            EXPECT_GT(model.loadings_norm()[i], 0.0);
        }
        EXPECT_EQ(static_cast<int>(model.loadings_norm().size()), 2);
    });
}

// joint selection of smoothing level and rank by GCV: the selection lies on the grids and recovers the signal
TEST(fpca, missing_data_gcv_selection) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;
    matrix_t Y = remove_entries(make_fpca_data(D, 50, 0.1, signal), 0.2, 2711);
    std::vector<double> grid = log_grid(-4, 0, 1);
    with_fpca(D, Y, [&](auto& model) {
        model.fit(4, grid, ComputeXactSVD | OptimizeGCV);
        EXPECT_EQ(model.gcv_scores().rows(), static_cast<int>(grid.size()));
        EXPECT_EQ(model.gcv_scores().cols(), 4);   // one column per rank
        EXPECT_TRUE(model.gcv_scores().array().isFinite().all());
        EXPECT_GE(model.F().cols(), 1);
        EXPECT_LE(model.F().cols(), 4);
        EXPECT_TRUE(std::find(grid.begin(), grid.end(), model.lambda()(0, 0)) != grid.end());
        EXPECT_LT(reconstruction_error(model, signal), 0.05);
    });
}

// k-fold cross-validation is the default calibration of missing-data fits when a grid is given
TEST(fpca, missing_data_kfold_default) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;
    matrix_t Y = remove_entries(make_fpca_data(D, 50, 0.1, signal), 0.2, 2711);
    std::vector<double> grid = log_grid(-4, 0, 1);
    with_fpca(D, Y, [&](auto& model) {
        model.fit(4, grid, ComputeXactSVD);   // no calibration flag
        EXPECT_EQ(model.gcv_scores().rows(), static_cast<int>(grid.size()));
        EXPECT_EQ(model.gcv_scores().cols(), 1);   // CV error along the grid, at the selected rank
        EXPECT_TRUE(model.gcv_scores().array().isFinite().all());
        EXPECT_LE(model.F().cols(), 4);
        EXPECT_LT(reconstruction_error(model, signal), 0.05);
    });
}

// the k-fold cross-validation runs its (lambda, fold) pairs in parallel when the MM scheme is given a factory of inner
// solvers: same CV table, selection and fit as the sequential run (exact SVD: the computation is deterministic)
TEST(fpca, missing_data_kfold_parallel_matches_sequential) {
    ASSERT_GT(parallel_get_num_threads(), 1) << "the parallel path needs more than one worker";
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;
    matrix_t Y = remove_entries(make_fpca_data(D, 50, 0.1, signal), 0.2, 2711);   // n_nodes x n_units
    std::vector<double> grid = log_grid(-4, 0, 1);
    FeSpace Vh(D, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(u), grad(v)));
    ZeroField<2> zero;
    auto L = integral(D)(zero * v);
    GeoFrame data(D);
    auto& layer = data.template insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", Y);
    internals::fe_ls_elliptic smoother;
    smoother.discretize(fe_ls_elliptic{a, L}.get());
    smoother.analyze_data(data, vector_t::Ones(data[0].rows()).asDiagonal());
    auto solver = fpca_subspace_solver().get(smoother);
    const int flag = ComputeXactSVD | OptimizeMSRE;
    // sequential: no factory
    internals::fpca_na_impl sequential(solver, smoother);
    sequential.fit(matrix_t(Y.transpose()), 4, grid, flag);
    // parallel: one inner solver per (lambda, fold) pair, on a copy of the smoother
    internals::fpca_na_impl parallel(solver, smoother, [](internals::fe_ls_elliptic& s) {
        return fpca_subspace_solver().get(s);
    });
    parallel.fit(matrix_t(Y.transpose()), 4, grid, flag);
    EXPECT_EQ(parallel.gcv_scores().rows(), static_cast<int>(grid.size()));
    EXPECT_EQ((parallel.gcv_scores() - sequential.gcv_scores()).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ(parallel.lambda()(0, 0), sequential.lambda()(0, 0));
    ASSERT_EQ(parallel.loading().cols(), sequential.loading().cols());
    EXPECT_EQ((parallel.loading() - sequential.loading()).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((parallel.scores() - sequential.scores()).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((parallel.center() - sequential.center()).cwiseAbs().maxCoeff(), 0.0);
}

// complete data: no mean unless ComputeMean is set; with ComputeMean the smooth mean is recovered
TEST(fpca, compute_mean_flag) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    const matrix_t& nodes = D.nodes();
    matrix_t signal;
    matrix_t Y = make_fpca_data(D, 50, 0.1, signal);
    vector_t mean_field(nodes.rows());
    for (int i = 0; i < nodes.rows(); ++i) { mean_field[i] = 2.0 + nodes(i, 0) * nodes(i, 1); }
    Y.colwise() += mean_field;
    std::vector<double> grid = log_grid(-4, 0, 1);
    with_fpca(D, Y, [&](auto& model) {
        model.fit(3, grid, ComputeXactSVD);
        EXPECT_EQ(model.center().norm(), 0.0);   // default: no centering
    });
    with_fpca(D, Y, [&](auto& model) {
        model.fit(3, grid, ComputeXactSVD | ComputeMean);
        EXPECT_LT((model.center_locs() - mean_field).norm() / mean_field.norm(), 0.05);
        const matrix_t& centred_signal = signal;   // the scores have zero mean
        matrix_t Yhat = model.Fn() * model.S().transpose();
        EXPECT_LT((centred_signal - Yhat).squaredNorm() / centred_signal.squaredNorm(), 0.1);
    });
}

// functional singular value thresholding on partially observed data (smoke test)
TEST(fpca, fsvt_missing_data) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    matrix_t signal;
    matrix_t Y = remove_entries(make_fpca_data(D, 50, 0.1, signal), 0.2, 2711);
    FeSpace Vh(D, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(u), grad(v)));
    ZeroField<2> zero;
    auto L = integral(D)(zero * v);
    GeoFrame data(D);
    auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", Y);
    fSVT model("y", data, fe_ls_elliptic(a, L));
    model.fit(/* threshold = */ 1.0, /* max_rank = */ 6, std::vector<double> {1e-2}, ComputeXactSVD);
    EXPECT_GE(model.rank(), 1);
    EXPECT_LE(model.rank(), 6);
    EXPECT_TRUE(model.U().array().isFinite().all() && model.V().array().isFinite().all());
}

// smooth mean of partially observed curves (smoke test)
TEST(fpca, frpde_missing_data) {
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(15);
    const matrix_t& nodes = D.nodes();
    matrix_t signal;
    matrix_t Y = make_fpca_data(D, 50, 0.1, signal);
    vector_t mean_field(nodes.rows());
    for (int i = 0; i < nodes.rows(); ++i) { mean_field[i] = 2.0 + nodes(i, 0) * nodes(i, 1); }
    Y.colwise() += mean_field;
    Y = remove_entries(Y, 0.2, 2711);
    FeSpace Vh(D, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(u), grad(v)));
    ZeroField<2> zero;
    auto L = integral(D)(zero * v);
    GeoFrame data(D);
    auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("Y", Y);
    FRPDE model("Y ~ f", data, fe_ls_elliptic(a, L));
    model.fit(1e-2);
    vector_t fitted = model.fitted();
    EXPECT_TRUE(fitted.array().isFinite().all());
    EXPECT_LT((fitted - mean_field).norm() / mean_field.norm(), 0.1);
}
