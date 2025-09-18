//
// Created by Marco Galliani on 21/05/25.
//
#include <fdaPDE/models.h>
using namespace fdapde;

#include "nlohmann/json.hpp"
using nlohmann::json;

#include <variant>

using fpca_solver_variant = std::variant<
    fpca_power_solver,
    fpca_subspace_solver,
    fpca_subspace_experimental_solver,
    fpca_direct_solver
>;

fpca_solver_variant get_fpca_solver(const std::string& solver_name) {
    if (solver_name == "subspace") return fpca_subspace_solver();
    else if (solver_name == "sequential") return fpca_power_solver();
    else if (solver_name == "subspace_experimental") return fpca_subspace_experimental_solver();
    else if (solver_name == "direct") return fpca_direct_solver();
    else throw std::invalid_argument("Unknown solver: " + solver_name);
}

int main() {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
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
    int n_units = X.rows(), n_locs = X.cols();
    matrix_t locs = read_csv<double>(data_path + "/locs.csv").as_matrix(); // (n_locs,2)-matrix
    // load the geoframe
    GeoFrame data(D);
    auto& l_locs = data.insert_scalar_layer<POINT>("locs_layer", locs);

    // center the data
    l_locs.load_vec("sample_mean", X.colwise().mean());
    SRPDE center("sample_mean ~ f", data, fe_ls_elliptic(a, F));
    // calibration
    std::vector<double> center_grid;
    for (double x = -7.0; x <= 2.0; x += 0.5) { center_grid.push_back(std::pow(10, x) / data[0].rows()); }
    GridOptimizer<1> opt;
    opt.optimize(center.gcv(), center_grid);
    center.fit(opt.optimum());

    // remove the smooth mean
    matrix_t X_c = X.rowwise() - center.fitted().transpose();

    //fit the model on the centred data
    l_locs.load_blk("X", X_c.transpose());
    fPCA fpca("X", data, fe_ls_elliptic(a, F));

    std::vector<double> lambda_grid = json_file.at("RunParams").at("lambda_grid").get<std::vector<double>>();
    //select the fpca solver according to the params.json
    std::visit(
        [&](auto&& solver){
            fpca.fit(
                /* n_comp = */ json_file["RunParams"].value("n_pc",6),
                lambda_grid,
                /* options = */ ComputeRandSVD | 0x1 | DoNotComputeMean, //run with no calibration
                solver
            );
        },
        get_fpca_solver(json_file["RunParams"].value("fpca_solver","subspace"))
    );

    // results
    std::string results_path = "test-results/";
    //(1) functional mean ()
    write_csv(results_path + "center.csv", center.f()); //at nodes
    write_csv(results_path + "center_locs.csv", center.fitted()); //at locs
    //(2) functional PCs
    write_csv(results_path + "loadings.csv", fpca.F()); //at nodes
    write_csv(results_path + "loadings_locs.csv", fpca.Fn()); //at locs
    //(3) scores
    write_csv(results_path + "scores.csv", fpca.S());
    //(4) reconstructed functional data
    matrix_t rec_X = fpca.S()*fpca.F().transpose();
    rec_X.rowwise() += center.f().transpose();
    write_csv(results_path + "reconstruction.csv", rec_X); //at nodes
    matrix_t rec_X_locs = fpca.S()*fpca.Fn().transpose();
    rec_X_locs.rowwise() += center.fitted().transpose();
    write_csv(results_path + "reconstruction_at_locs.csv", rec_X_locs); //at locs
    //(6) lambda
    write_csv(results_path + "lambda.csv", fpca.lambda());

    return 0;
}