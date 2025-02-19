//
// Created by Marco Galliani on 25/01/25.
//
#include "fdaPDE/models/functional/fpca.h"
using fdapde::models::SRPDE;
using fdapde::models::STRPDE;

#include <Eigen/Cholesky>

#include <fstream>
#include <iostream>
#include <cstddef>
#include <chrono>

#include "fdaPDE/core.h"
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;


#include <Eigen/SVD>

#include "fdaPDE/models/functional/fpca.h"
#include "fdaPDE/models/sampling_design.h"
using fdapde::models::FPCA;
using fdapde::models::RegularizedSVD;
using fdapde::models::Sampling;
using fdapde::models::SubspaceIteration;

#include "fdaPDE/calibration/symbols.h"
#include "fdaPDE/models/regression/stochastic_edf.h"
using fdapde::calibration::Calibration;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;

#include "test/src/utils/constants.h"
#include "test/src/utils/mesh_loader.h"
#include "test/src/utils/utils.h"
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

using fdapde::almost_equal;


void fpca_subspace_test(DVector<double> lambda_grid, int n_pcs=3){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define the regularized svd solver
    RegularizedSVD<fdapde::monolithic,RSVD<DMatrix<double>>> rsvd;

    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_lambda_D(lambda_grid(0));
    model.set_npc(n_pcs);
    model.set_spatial_locations(locs);
    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    const auto start{std::chrono::steady_clock::now()};
    model.init();

    //init step
    RSVD<DMatrix<double>> svd;
    svd.compute(y,3);
    // subspace iterations
    fdapde::models::SubspaceIteration<decltype(model)> subspace_solver(model, 1e-6, 150, 1412);

    DMatrix<double> selected_lambdas(3,1);
    for(int k=1; k<=3; k++){
        ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
            selected_lambdas.row(k-1) = lambda;
            subspace_solver.compute(model.X(), k, selected_lambdas.topRows(k), svd.matrixV().leftCols(k));
            return subspace_solver.gcv();   // return GCV index at convergence
        });
        selected_lambdas.row(k-1) = fdapde::core::Grid<Dynamic>{}.optimize(gcv, lambda_grid);
    }
    const auto end{std::chrono::steady_clock::now()};
    subspace_solver.compute(model.X(), 3, selected_lambdas.topRows(3), svd.matrixV().leftCols(3));

    Eigen::saveMarket(subspace_solver.Fn(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(subspace_solver.S(),"results/estimated_Scores.mtx");
    std::ofstream exe_times_file("results/execution_times.csv");
    exe_times_file << (std::chrono::duration<double>{end - start}).count() << std::endl;
    exe_times_file.close();

    std::cout << selected_lambdas << std::endl;
}

void fpca_subspace_fixed(double fixed_lambda, int max_iter, int n_pcs=3){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define the regularized svd solver
    RegularizedSVD<fdapde::monolithic,RSVD<DMatrix<double>>> rsvd;

    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_lambda_D(fixed_lambda);
    model.set_npc(n_pcs);
    model.set_spatial_locations(locs);
    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    const auto start{std::chrono::steady_clock::now()};
    model.init();

    //init step
    RSVD<DMatrix<double>> svd;
    svd.compute(y,3);
    // subspace iterations
    fdapde::models::FixedLambdaSubspaceIteration<decltype(model)> subspace_solver(model, 1e-6, 150, 1412);
    subspace_solver.compute(model.X(), 3, DVector<double>::Constant(1,fixed_lambda), svd.matrixV().leftCols(3));
    const auto end{std::chrono::steady_clock::now()};

    Eigen::saveMarket(subspace_solver.Fn(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(subspace_solver.S(),"results/estimated_Scores.mtx");
    std::ofstream exe_times_file("results/execution_times.csv");
    exe_times_file << (std::chrono::duration<double>{end - start}).count() << std::endl;
    exe_times_file.close();
}

void fpca_subspace_fixed_calibration(DVector<double> lambda_grid, int n_pcs=3){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define the regularized svd solver
    RegularizedSVD<fdapde::monolithic,RSVD<DMatrix<double>>> rsvd;

    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_npc(n_pcs);
    model.set_spatial_locations(locs);
    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    const auto start{std::chrono::steady_clock::now()};
    model.init();
    //init step
    RSVD<DMatrix<double>> svd;
    svd.compute(y,3);

    //(1) Initialise SubspaceIterations solver
    fdapde::models::FixedLambdaSubspaceIteration<decltype(model)> subspace_solver(model, 1e-6, 150, 1412);

    //(2) Select the optimal \lambda
    ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
        subspace_solver.compute(model.X(), 3, lambda, svd.matrixV().leftCols(3));
        return subspace_solver.gcv(model.X());   // return GCV index at convergence
    });
    DVector<double> optimal_lambda = fdapde::core::Grid<Dynamic>{}.optimize(gcv, lambda_grid);
    //(3) Run with the optimal lambda
    subspace_solver.compute(model.X(), 3, optimal_lambda, svd.matrixV().leftCols(3));

    Eigen::saveMarket(subspace_solver.Fn(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(subspace_solver.S(),"results/estimated_Scores.mtx");
}