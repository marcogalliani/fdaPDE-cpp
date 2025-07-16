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
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;

    std::ifstream input("params.json");
    json json_file = json::parse(input);

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
    binary_t nan_pattern = na_matrix(X);
    int n_units = X.rows(), n_locs = X.cols();
    matrix_t locs = read_csv<double>(data_path + "/locs.csv").as_matrix(); // (n_locs,2)-matrix

    //fPCA
    sparse_matrix_t Psi = internals::point_basis_eval(Vh, locs);
    GeoFrame data(D);
    auto& l_locs = data.insert_scalar_layer<POINT>("locs_layer", locs);
    l_locs.load_blk("X", X.transpose());

    fPCA fpca("X", data, fe_ls_elliptic(a, F));

    std::vector<double> lambda_grid = json_file.at("RunParams").at("lambda_grid").get<std::vector<double>>();
    fpca.fit(
        /* n_comp = */ json_file["RunParams"].value("n_pc",6),
        lambda_grid,
        /* options = */ ComputeRandSVD | OptimizeGCV,
        fpca_subspace_experimental_solver()
    );

    // save results
    std::string results_path = "test-results/";
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