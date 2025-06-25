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
    std::ifstream input("params.json");
    json json_file = json::parse(input);

    // geometry
    // -> space
    std::string mesh_path = json_file["Path"].value("mesh","/Users/marcogalliani/Projects/fpca-simulations/simulations/data/mesh");
    mesh_path = std::filesystem::relative(mesh_path, std::filesystem::current_path());

    Triangulation<2, 2> D(mesh_path + "/points.csv", mesh_path + "/elements.csv", mesh_path + "/boundary.csv",
        /* header = */ true, /* index_col = */ true);
    // -> time
    Triangulation<1, 1> T(mesh_path + "/time_grid.csv",
        /* header = */ true, /* index_col = */ true);

    // data
    std::string data_path = json_file["Path"].value("data","/Users/marcogalliani/Projects/fpca-simulations/simulations/data/mesh");
    data_path = std::filesystem::relative(data_path, std::filesystem::current_path());

    GeoFrame data(D,T);
    auto& l = data.insert_scalar_layer<POINT,POINT>("layer", std::pair {data_path + "/locs.csv", data_path + "/times.csv"});
    // data have to be organised in a (n_locs, n_units)-matrix
    Eigen::Matrix<double, Dynamic, Dynamic> X = read_csv<double>(data_path + "/y.csv").as_matrix().transpose();
    // to load an eigen matrix
    for(int i = 0; i < X.cols(); ++i) { l.load_vec("x" + std::to_string(i + 1), X.col(i)); }
    l.data().merge<double>("X");

    // physics
    // -> space: isotropic laplacian
    FeSpace Vh(D, P1<1>);
    TrialFunction f_D(Vh);
    TestFunction  v_D(Vh);
    auto a_D = integral(D)(dot(grad(f_D), grad(v_D)));
    ZeroField<2> u_D;
    auto F_D = integral(D)(u_D * v_D);
    // -> time
    BsSpace Bh(T, 3);   // cubic B-splines in time
    // trial and test function definition
    TrialFunction f_t(Bh);
    TestFunction  v_t(Bh);
    //
    auto a_t = integral(T)(dxx(f_t)*dxx(v_t));
    ZeroField<1> u_t;
    auto F_t = integral(T)(u_t * v_t);

    // model
    fPCA m("X", data,
        fe_ls_separable_mono(std::pair {a_D, F_D}, std::pair {a_t, F_t}));

    std::vector<double> lambda_grid = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2};
    m.fit(
        /* n_comp = */ 5,
        lambda_grid,
        /* options = */ OptimizeMSRE | ComputeRandSVD,
        fpca_subspace_solver()
    );

    std::string results_path = json_file["Path"].value("results","/Users/marcogalliani/Projects/fpca-simulations/simulations/lambda-sensibility/cpp-script/results");

    write_csv(results_path + "lambda.csv", m.lambda()); // m.lambda() not working: lambda_ has to be passed from the solver
    write_csv(results_path + "loadings.csv", m.Fn());
    write_csv(results_path + "scores.csv", m.S());
    write_csv(results_path + "y_reconstructed.csv", m.S()*m.Fn().transpose());

    return 0;
}