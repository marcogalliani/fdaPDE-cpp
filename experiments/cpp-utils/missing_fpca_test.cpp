//
// Created by Marco Galliani on 08/11/24.
//

#include "fdaPDE/core.h"
using fdapde::core::FEM;
using fdapde::core::SPLINE;
using fdapde::core::spline_order;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::bilaplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

#include "fdaPDE/models/functional/fpca.h"
using fdapde::models::FPCA;
using fdapde::models::RegularizedSVD;
using fdapde::models::Sampling;
using fdapde::models::SpaceTimeSeparable;

#include "test/src/utils/constants.h"
#include "test/src/utils/mesh_loader.h"
#include "test/src/utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::write_csv;
using fdapde::testing::read_mtx;


template<typename RSVDPolicy_, typename SVDPolicy_=RSVD<DMatrix<double>>>
void fpca_miss_test(double fixed_lambda, int n_pc, int start_pt=1){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_miss_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);

    // define the regularized svd solver
    RegularizedSVD<RSVDPolicy_,SVDPolicy_,true> rsvd;
    rsvd.set_start(start_pt);

    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_spatial_locations(locs);
    model.set_lambda_D(fixed_lambda);
    model.set_npc(n_pc);

    //load the data
    DMatrix<double> X = read_csv<double>("../data/models/fpca/2D_miss_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, X);
    model.set_data(df);

    const auto start{std::chrono::steady_clock::now()};
    model.init();
    model.solve();
    const auto end{std::chrono::steady_clock::now()};

    DMatrix<double> exe_time = DMatrix<double>::Constant(1,1,(std::chrono::duration<double>{end - start}).count());

    //Results
    Eigen::saveMarket(model.Psi() * model.loadings(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(model.scores(),"results/estimated_Scores.mtx");
    write_csv("results/execution_time.csv", exe_time);
}

template<typename RSVDPolicy_, typename SVDPolicy_=RSVD<DMatrix<double>>>
void st_fpca_miss_test(double lambda_D, double lambda_T, int t_steps){
    // define time domain
    Triangulation<1, 1> time_mesh(0, 1, t_steps);
    // define domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2d_miss_test/datamatrix_centred.csv");
    // define regularizing PDE in space
    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);
    // define the regularized svd solver
    RegularizedSVD<RSVDPolicy_,SVDPolicy_,true> rsvd;
    //define the model
    FPCA<SpaceTimeSeparable> model(
            space_penalty, time_penalty, Sampling::mesh_nodes, rsvd);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    // solve smoothing problem
    const auto start{std::chrono::steady_clock::now()};
    model.init();
    model.solve();
    const auto end{std::chrono::steady_clock::now()};

    DMatrix<double> exe_time = DMatrix<double>::Constant(1,1,(std::chrono::duration<double>{end - start}).count());

    //Results
    Eigen::saveMarket(model.Psi() * model.loadings(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(model.scores(),"results/estimated_Scores.mtx");
    write_csv("results/execution_time.csv", exe_time);

    Eigen::saveMarket(model.R0(),"results/R0.mtx");
}


