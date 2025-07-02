//
// Created by Marco Galliani on 21/05/25.
//
#include <fdaPDE/models.h>
using namespace fdapde;

#include "nlohmann/json.hpp"
using nlohmann::json;

int main() {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;

    std::ifstream input("params.json");
    json json_file = json::parse(input);
    std::string results_path = json_file["Path"].value("results","/Users/marcogalliani/Projects/fpca-simulations/simulations/lambda-sensibility/cpp-script/results");

    // geometry
    std::string mesh_path = json_file["Path"].value("mesh","/Users/marcogalliani/Projects/fpca-simulations/simulations/data/mesh");
    mesh_path = std::filesystem::relative(mesh_path, std::filesystem::current_path());

    Triangulation<2, 2> D(mesh_path + "/points.csv", mesh_path + "/elements.csv", mesh_path + "/boundary.csv",
    /* header = */ true, /* index_col = */ true);

    // physics (isotropic laplacian)
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);

    // data
    std::string data_path = json_file["Path"].value("data","/Users/marcogalliani/Projects/fpca-simulations/simulations/data/mesh");
    data_path = std::filesystem::relative(data_path, std::filesystem::current_path());
    // load the data matrix (assume that X is an (n_units,n_locs) data matrix)
    matrix_t X = read_csv<double>(data_path + "/y.csv").as_matrix();
    int n_units = X.rows(), n_locs = X.cols();
    matrix_t locs = read_csv<double>(data_path + "/locs.csv").as_matrix(); // (n_locs,2)-matrix
    /*
    vector_t vec_X = Eigen::Map<vector_t>(X.data(), X.size());
    //now we have to compute the matrix of locations at which we have available observations
    matrix_t repeated_locs = kronecker(locs, vector_t::Ones(n_units));

    //we are ready to store everything in the dataframe
    GeoFrame data(D);
    auto& l_repeated_locs = data.insert_scalar_layer<POINT>("rep_locs_layer", repeated_locs);
    l_repeated_locs.load_vec("y",vec_X);

    //(1) compute the functional mean (with missing data)
    SRPDE f_mean("y ~ f", data, fe_ls_elliptic(a, F));

    // calibration
    std::vector<double> center_grid;
    for (double x = -8.0; x <= 0.0; x += 0.5) { center_grid.push_back(std::pow(10, x)); }
    GridOptimizer<1> opt;
    opt.optimize(f_mean.gcv(), center_grid);

    // fit at optimal smoothing level
    std::cout << opt.optimum()[0] << std::endl;
    f_mean.fit(opt.optimum());

    write_csv(results_path + "center.csv", f_mean.f());
    write_csv(results_path + "center_locs.csv", f_mean.fitted().head(n_locs));


    //(2) fPCA
    //compute and load the centred data
    matrix_t X_c = X.rowwise() - f_mean.fitted().head(n_locs).transpose();
    */
    binary_t nan_pattern = na_matrix(X);
    sparse_matrix_t Psi = internals::point_basis_eval(Vh, locs);
    GeoFrame centred_data(D); //unfortunately we need a new GeoFrame
    auto& l_locs = centred_data.insert_scalar_layer<POINT>("locs_layer", locs);
    //l_locs.load_blk("X", X_c.transpose());
    l_locs.load_blk("X", X.transpose());

    fPCA fpca("X", centred_data, fe_ls_elliptic(a, F));

    std::vector<double> lambda_grid = json_file.at("RunParams").at("lambda_grid").get<std::vector<double>>();
    fpca.fit(
        /* n_comp = */ json_file["RunParams"].value("n_pc",6),
        lambda_grid,
        /* options = */ ComputeRandSVD | OptimizeMSRE,
        fpca_subspace_solver()
    );


    // save results
    //(1) functional mean ()
    write_csv(results_path + "center.csv", fpca.center()); //at nodes
    write_csv(results_path + "center_locs.csv", Psi*fpca.center()); //at locs
    //(2) functional PCs
    write_csv(results_path + "loadings.csv", fpca.F()); //at nodes
    write_csv(results_path + "loadings_locs.csv", fpca.Fn()); //at locs
    //(3) scores
    write_csv(results_path + "scores.csv", fpca.S());
    //(4) reconstructed functional data
    matrix_t rec_X = fpca.S()*fpca.F().transpose();
    rec_X = rec_X.rowwise() + fpca.center().transpose();
    write_csv(results_path + "reconstruction.csv", rec_X); //at nodes
    matrix_t rec_X_locs = fpca.S()*fpca.Fn().transpose();
    rec_X_locs = rec_X_locs.rowwise() + (Psi*fpca.center()).transpose();
    write_csv(results_path + "reconstruction_at_locs.csv", rec_X_locs); //at locs
    //(5) imputation
    matrix_t X_imputed = (~nan_pattern).select(X,0) + (nan_pattern).select(rec_X_locs,0);
    write_csv(results_path + "imputation.csv", X_imputed);
    //(6) lambda
    write_csv(results_path + "lambda.csv", fpca.lambda());

    return 0;
}