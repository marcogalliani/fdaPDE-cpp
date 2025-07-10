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

    //GapFiller
    sparse_matrix_t Psi = internals::point_basis_eval(Vh, locs);
    GeoFrame data(D);
    auto& l_locs = data.insert_scalar_layer<POINT>("locs_layer", locs);
    l_locs.load_blk("X", X.transpose());

    GapFiller gap_filler("X", data, fe_ls_elliptic(a, F));

    std::vector<double> lambda_grid = json_file.at("RunParams").at("lambda_grid").get<std::vector<double>>();
    gap_filler.fit(
        lambda_grid,
        /* options = */ OptimizeGCV
    );

    // save results
    std::string results_path = "test-results/";
    // reconstruction
    write_csv(results_path + "lambda.csv", gap_filler.lambda());
    write_csv(results_path + "reconstruction.csv", gap_filler.U()); //at nodes
    write_csv(results_path + "reconstruction_at_locs.csv", gap_filler.U()*Psi.transpose()); //at locations


    return 0;
}