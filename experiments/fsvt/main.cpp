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

    // Geometry
    std::string mesh_path = json_file["Path"].value("mesh","/Users/marcogalliani/Projects/fpca-simulations/simulations/data/mesh");
    mesh_path = std::filesystem::relative(mesh_path, std::filesystem::current_path());
    Triangulation<2, 2> D(mesh_path + "/points.csv", mesh_path + "/elements.csv", mesh_path + "/boundary.csv",
    /* header = */ true, /* index_col = */ true);

    // Physics (isotropic laplacian)
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);

    // Data
    std::string data_path = json_file["Path"].value("data","/Users/marcogalliani/Projects/fpca-simulations/simulations/data/mesh");
    data_path = std::filesystem::relative(data_path, std::filesystem::current_path());
    // load the data matrix (assume that X is an (n_units,n_locs) data matrix)
    matrix_t X = read_csv<double>(data_path + "/y.csv").as_matrix();
    int n_units = X.rows(), n_locs = X.cols();
    matrix_t locs = read_csv<double>(data_path + "/locs.csv").as_matrix(); // (n_locs,2)-matrix
    binary_t nan_pattern = na_matrix(X);
    // load in GeoFrame
    GeoFrame data(D);
    auto& l_locs = data.insert_scalar_layer<POINT>("locs_layer", locs);
    //l_locs.load_blk("X", X_c.transpose());
    l_locs.load_blk("X", X.transpose());

    // Model
    fSVT fsvt("X", data, fe_ls_elliptic(a, F));
    // fit
    std::vector<double> lambda_grid = json_file.at("RunParams").at("lambda_grid").get<std::vector<double>>();
    fsvt.fit(
        /* threshold = */ json_file["RunParams"].value("sing_val_th",0.1),
        /* maximum rank = */ json_file["RunParams"].value("max_rank",100),
        lambda_grid,
        /* options = */ ComputeRandSVD,
        fpca_subspace_solver()
    );
    // Results
    std::string results_path = "test-results/";

    write_csv(results_path + "singular_values.csv", fsvt.singularValues());
    write_csv(results_path + "left_singular_vectors.csv", fsvt.U());
    write_csv(results_path + "right_singular_functions.csv", fsvt.V());
    write_csv(results_path + "right_singular_functions_locs.csv", fsvt.Vn());

    matrix_t reconstruction = fsvt.U()*fsvt.singularValues().asDiagonal()*fsvt.V().transpose();
    write_csv(results_path + "reconstruction.csv", reconstruction);
    matrix_t reconstruction_locs = fsvt.U()*fsvt.singularValues().asDiagonal()*fsvt.Vn().transpose();
    write_csv(results_path + "reconstruction_locs.csv", reconstruction_locs);
    //(5) imputation
    matrix_t X_imputed = (~nan_pattern).select(X,0) + (nan_pattern).select(reconstruction_locs,0);
    write_csv(results_path + "imputation.csv", X_imputed);
    //(6) lambda
    write_csv(results_path + "lambda.csv", fsvt.lambda());


    return 0;
}