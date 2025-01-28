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

#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <iostream>
#include <cstddef>

#include <fdaPDE/core.h>
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::SPLINE;
using fdapde::core::spline_order;
using fdapde::core::laplacian;
using fdapde::core::bilaplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;


#include <Eigen/SVD>

#include "../../fdaPDE/models/functional/fpca.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::FPCA;
using fdapde::models::RegularizedSVD;
using fdapde::models::Sampling;
using fdapde::models::SpaceTimeSeparable;
#include "../../fdaPDE/calibration/symbols.h"
using fdapde::calibration::Calibration;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver:       sequential (power iteration)
//    SVD:          RSI
TEST(fpca_test, laplacian_samplingatnodes_sequential) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;
    RegularizedSVD<fdapde::sequential,RSVD<DMatrix<double>>> rsvd;
    rsvd.set_seed(1412);
    FPCA<SpaceOnly> model(pde, Sampling::mesh_nodes, rsvd);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();
    // test correctness (dealing with uniquess up to sign changes)
    SpMatrix<double> mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test1/loadings_seq.mtx");
    DMatrix<double> true_loadings = mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test1/scores_seq.mtx");
    DMatrix<double> true_scores = mem_buff;
    bool equality_PCs = true;
    bool equality_Scores = true;
    for(int i=0; i < 3 && equality_PCs; i++){
        equality_PCs = almost_equal(model.Psi() * model.loadings().col(i),true_loadings.col(i)) ||
                       almost_equal(-model.Psi() * model.loadings().col(i),true_loadings.col(i));
    }
    for(int i=0; i < 3 && equality_Scores; i++){
        equality_Scores = almost_equal(model.scores().col(i),true_scores.col(i)) ||
                          almost_equal(-model.scores().col(i),true_scores.col(i));
    }
    EXPECT_TRUE(equality_PCs);
    EXPECT_TRUE(equality_Scores);
}

// test 2
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1af
//    missing data: no
//    solver:       monolithic (rsvd)
//    SVD:          RSI
TEST(fpca_test, laplacian_samplingatnodes_monolithic) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;

    RegularizedSVD<fdapde::monolithic,RSVD<DMatrix<double>>> rsvd;
    rsvd.set_seed(1412);

    FPCA<SpaceOnly> model(problem, Sampling::mesh_nodes,rsvd);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();

    // test correctness (dealing with uniquess up to sign changes)
    SpMatrix<double> mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test1/loadings_mon.mtx");
    DMatrix<double> true_loadings = mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test1/scores_mon.mtx");
    DMatrix<double> true_scores = mem_buff;
    bool equality_PCs = true;
    bool equality_Scores = true;
    for(int i=0; i < 3 && equality_PCs; i++){
        equality_PCs = almost_equal(model.Psi() * model.loadings().col(i),true_loadings.col(i)) ||
                almost_equal(-model.Psi() * model.loadings().col(i),true_loadings.col(i));
    }
    for(int i=0; i < 3 && equality_Scores; i++){
        equality_Scores = almost_equal(model.scores().col(i),true_scores.col(i)) ||
                         almost_equal(-model.scores().col(i),true_scores.col(i));
    }
    EXPECT_TRUE(equality_PCs);
    EXPECT_TRUE(equality_Scores);
}

// test 3
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver: sequential (power iteration) + GCV \lambda selection
//TEST(fpca_test, laplacian_samplingatlocations_sequential_gcv) {
//    // define domain
//    MeshLoader<Triangulation<2, 2>> domain("unit_square");
//    // import data from files
//    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test2/locs.csv");
//    DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test2/y.csv");
//    // define regularizing PDE
//    auto L = -laplacian<FEM>();
//    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
//    // grid of smoothing parameters
//    DMatrix<double> lambda_grid(20, 1);
//    for (int i = 0; i < 20; ++i) lambda_grid(i, 0) = std::pow(10, -4 + 0.1 * i);
//    // define model
//    RegularizedSVD<fdapde::sequential,Eigen::JacobiSVD<DMatrix<double>>> rsvd(Calibration::gcv);
//    rsvd.set_lambda(lambda_grid);
//    rsvd.set_seed(78965);   // for reproducibility purposes in testing
//    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
//    model.set_spatial_locations(locs);
//    // set model's data
//    BlockFrame<double, int> df;
//    df.insert(OBSERVATIONS_BLK, y);
//    model.set_data(df);
//    // solve FPCA problem
//    model.init();
//    model.solve();
//    // test correctness
//    EXPECT_TRUE(almost_equal(model.Psi() * model.loadings(), "../data/models/fpca/2D_test2/loadings_seq.mtx"));
//    EXPECT_TRUE(almost_equal(model.scores(),                 "../data/models/fpca/2D_test2/scores_seq.mtx"  ));
//}

// test 4
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver: sequential (power iteration) + KCV \lambda selection
//TEST(fpca_test, laplacian_samplingatlocations_sequential_kcv) {
//    // define domain
//    MeshLoader<Triangulation<2, 2>> domain("unit_square");
//    // import data from files
//    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test3/locs.csv");
//    DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test3/y.csv");
//    // define regularizing PDE
//    auto L = -laplacian<FEM>();
//    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//    // grid of smoothing parameters
//    DMatrix<double> lambda_grid(20, 1);
//    for (int i = 0; i < 20; ++i) lambda_grid(i, 0) = std::pow(10, -4 + 0.1 * i);
//    // define model
//    RegularizedSVD<fdapde::sequential,Eigen::JacobiSVD<DMatrix<double>>> rsvd(Calibration::kcv);
//    rsvd.set_lambda(lambda_grid);
//    rsvd.set_seed(12654);   // for reproducibility purposes in testing
//    FPCA<SpaceOnly> model(problem, Sampling::pointwise, rsvd);
//    model.set_spatial_locations(locs);
//    // set model's data
//    BlockFrame<double, int> df;
//    df.insert(OBSERVATIONS_BLK, y);
//    model.set_data(df);
//    // solve FPCA problem
//    model.init();
//    model.solve();
//    // test correctness
//    EXPECT_TRUE(almost_equal(model.Psi() * model.loadings(), "../data/models/fpca/2D_test3/loadings_seq.mtx"));
//    EXPECT_TRUE(almost_equal(model.scores(),                 "../data/models/fpca/2D_test3/scores_seq.mtx"  ));
//}

/*
// test 5
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations == nodes
//    penalization: space-time separable
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver: sequential (power iteration)
TEST(fpca_test, laplacian_samplingatnodes_separable_sequential) {
    // define time domain
    Triangulation<1, 1> time_mesh(0, 1, 14);
    // define domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("unit_square15");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test5/y.csv");
    // define regularizing PDE in space
    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);
     // define model
    double lambda_D = std::pow(10, -3.6); // 1e-3.6
    double lambda_T = std::pow(10, -2.2); // 1e-2.2
    //
    RegularizedSVD<fdapde::sequential,RSVD<DMatrix<double>>> rsvd;

    FPCA<SpaceTimeSeparable> model(
     space_penalty, time_penalty, Sampling::mesh_nodes, rsvd);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
}
*/

// test 4
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: yes
//    solver:       sequential
TEST(fpca_test, laplacian_samplingatnodes_nocalibration_missingdata_sequential) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test_missing/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;

    RegularizedSVD<fdapde::sequential,RSVD<DMatrix<double>>,true> rsvd;
    rsvd.set_seed(1412);
    FPCA<SpaceOnly> model(problem, Sampling::mesh_nodes, rsvd);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();

    // test correctness (dealing with uniquess up to sign changes)
    SpMatrix<double> mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test_missing/loadings_seq.mtx");
    DMatrix<double> true_loadings = mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test_missing/scores_seq.mtx");
    DMatrix<double> true_scores = mem_buff;
    bool equality_PCs = true;
    bool equality_Scores = true;
    for(int i=0; i < 3 && equality_PCs; i++){
        equality_PCs = almost_equal(model.Psi() * model.loadings().col(i),true_loadings.col(i)) ||
                       almost_equal(-model.Psi() * model.loadings().col(i),true_loadings.col(i));
    }
    for(int i=0; i < 3 && equality_Scores; i++){
        equality_Scores = almost_equal(model.scores().col(i),true_scores.col(i)) ||
                          almost_equal(-model.scores().col(i),true_scores.col(i));
    }
    EXPECT_TRUE(equality_PCs);
    EXPECT_TRUE(equality_Scores);
}

// test 5
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: yes
//    solver:       monolithic
TEST(fpca_test, laplacian_samplingatnodes_nocalibration_missingdata_monolithic) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test_missing/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;

    RegularizedSVD<fdapde::monolithic,RSVD<DMatrix<double>>,true> rsvd;
    rsvd.set_seed(1412);
    FPCA<SpaceOnly> model(problem, Sampling::mesh_nodes, rsvd);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();

    // test correctness (dealing with uniquess up to sign changes)
    SpMatrix<double> mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test_missing/loadings_mono.mtx");
    DMatrix<double> true_loadings = mem_buff;
    Eigen::loadMarket(mem_buff, "../data/models/fpca/2D_test_missing/scores_mono.mtx");
    DMatrix<double> true_scores = mem_buff;
    bool equality_PCs = true;
    bool equality_Scores = true;
    for(int i=0; i < 3 && equality_PCs; i++){
        equality_PCs = almost_equal(model.Psi() * model.loadings().col(i),true_loadings.col(i)) ||
                       almost_equal(-model.Psi() * model.loadings().col(i),true_loadings.col(i));
    }
    for(int i=0; i < 3 && equality_Scores; i++){
        equality_Scores = almost_equal(model.scores().col(i),true_scores.col(i)) ||
                          almost_equal(-model.scores().col(i),true_scores.col(i));
    }
    EXPECT_TRUE(equality_PCs);
    EXPECT_TRUE(equality_Scores);
}


