//
// Created by Marco Galliani on 21/05/25.
//
#include <fdaPDE/models.h>
using namespace fdapde;

#include "nlohmann/json.hpp"
using nlohmann::json;

Eigen::MatrixXd align_columns_by_sign(const Eigen::MatrixXd& reference, Eigen::MatrixXd target) {
    for (int col = 0; col < reference.cols(); ++col) {
        double dot_product = reference.col(col).dot(target.col(col));
        if (dot_product < 0) {
            target.col(col) *= -1;
        }
    }
    return target;
}

int main() {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;

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

    write_csv(results_path + "lambda.csv", fpca.lambda());
    write_csv(results_path + "loadings_locs.csv", fpca.Fn());
    write_csv(results_path + "loadings.csv", fpca.F());
    write_csv(results_path + "scores.csv", fpca.S());
    write_csv(results_path + "y_reconstructed.csv", fpca.S()*fpca.Fn().transpose());

    return 0;
}