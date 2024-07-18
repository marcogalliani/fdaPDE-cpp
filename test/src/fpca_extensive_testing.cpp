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

#include <chrono>

#include <fdaPDE/core.h>
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::PDE;


#include <Eigen/SVD>

#include "../../fdaPDE/models/functional/fpca.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::FPCA;
using fdapde::models::RegularizedSVD;
using fdapde::models::Sampling;
#include "../../fdaPDE/calibration/symbols.h"
using fdapde::calibration::Calibration;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

template<typename SVDType, typename SolutionPolicy_>
void fpca_generated_test(SVDType svd){
    // define domain
    MeshLoader<Mesh2D> domain("unit_square_rsvd_test");

    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test_rsvd/locs.csv");
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test_rsvd/datamatrix_centred-1.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // grid of smoothing parameters
    DMatrix<double> lambda_grid(20, 1);
    for (int i = 0; i < 20; ++i) lambda_grid(i, 0) = std::pow(10, -4 + 0.1 * i);

    // rsvd solver
    RegularizedSVD<SolutionPolicy_,SVDType> rsvd(Calibration::gcv, svd);
    rsvd.set_lambda(lambda_grid);

    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    // solve FPCA problem
    const auto start{std::chrono::steady_clock::now()};
    model.init();
    model.solve();
    const auto end{std::chrono::steady_clock::now()};
    std::cout << "Elapsed_time: " << (std::chrono::duration<double>{end - start}).count() << std::endl;

    // Distance from the actual solution
    //-> PC functions
    DVector<double> f1 = read_csv<double>("../data/models/fpca/2D_test_rsvd/f1.csv");
    DVector<double> f2 = read_csv<double>("../data/models/fpca/2D_test_rsvd/f2.csv");
    DVector<double> f3 = read_csv<double>("../data/models/fpca/2D_test_rsvd/f3.csv");

    DMatrix<double> est_PCs = model.Psi() * model.loadings();

    Eigen::saveMarket(est_PCs.col(0), "../est_PC1.mtx");
    Eigen::saveMarket(est_PCs.col(1), "../est_PC2.mtx");
    Eigen::saveMarket(est_PCs.col(2), "../est_PC3.mtx");

    DMatrix<double> XYZ(f1.rows(), 2);
    XYZ.col(0) = f1;
    XYZ.col(1) = est_PCs.col(0);
    std::cout << XYZ << std::endl;


    //1st autofunction
    std::cout << "First PC" << std::endl;
    std::cout << (f1 - est_PCs.col(0)).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (f1 + est_PCs.col(0)).lpNorm<Eigen::Infinity>() << std::endl;
    //2nd autofunction
    std::cout << "Second PC" << std::endl;
    std::cout << (f2 - est_PCs.col(1)).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (f2 + est_PCs.col(1)).lpNorm<Eigen::Infinity>() << std::endl;
    //3rd autofunction
    std::cout << "Third PC" << std::endl;
    std::cout << (f3 - est_PCs.col(2)).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (f3 + est_PCs.col(2)).lpNorm<Eigen::Infinity>() << std::endl;

    //-> Scores
    DVector<double> scores_1 = read_csv<double>("../data/models/fpca/2D_test_rsvd/score1-1.csv");
    DVector<double> scores_2 = read_csv<double>("../data/models/fpca/2D_test_rsvd/score2-1.csv");
    DVector<double> scores_3 = read_csv<double>("../data/models/fpca/2D_test_rsvd/score3-1.csv");

    //1st PC scores
    std::cout << "First PC scores" << std::endl;
    std::cout << (scores_1 - model.scores().col(0)).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (scores_1 + model.scores().col(0)).lpNorm<Eigen::Infinity>() << std::endl;
    //2nd scores
    std::cout << "Second PC scores" << std::endl;
    std::cout << (scores_2 - model.scores().col(1)).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (scores_2 + model.scores().col(1)).lpNorm<Eigen::Infinity>() << std::endl;
    //3rd scores
    std::cout << "Third PC scores" << std::endl;
    std::cout << (scores_3 - model.scores().col(2)).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (scores_3 + model.scores().col(2)).lpNorm<Eigen::Infinity>() << std::endl;

    return;
}

template<typename SVDType, typename SolutionPolicy_>
void fpca_library_test(SVDType svd){
    // define domain
    MeshLoader<Mesh2D> domain("unit_square");

    // import data from files
    //DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test_rsvd/locs.csv");
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;
    FPCA<SpaceOnly> model(pde, Sampling::mesh_nodes, RegularizedSVD<SolutionPolicy_,SVDType>(svd));
    // model.set_spatial_locations(locs);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    // solve FPCA problem
    const auto start{std::chrono::steady_clock::now()};
    model.init();
    model.solve();
    const auto end{std::chrono::steady_clock::now()};
    std::cout << "Elapsed_time: " << (std::chrono::duration<double>{end - start}).count() << std::endl;

    // Distance from the actual solution
    //-> PC functions
    DMatrix<double> loadings = read_mtx<double>("../data/models/fpca/2D_test1/loadings_seq.mtx");
    DMatrix<double> est_loadings = model.Psi() * model.loadings();

    std::cout << "Loadings" << std::endl;
    std::cout << (loadings - est_loadings).colwise().lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (loadings + est_loadings).colwise().lpNorm<Eigen::Infinity>() << std::endl;

    //-> Scores
    DMatrix<double> scores = read_mtx<double>("../data/models/fpca/2D_test1/scores_seq.mtx");
    DMatrix<double> est_scores = model.scores();

    //1st PC scores
    std::cout << "Scores" << std::endl;
    std::cout << (scores - est_scores).colwise().lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (scores + est_scores).colwise().lpNorm<Eigen::Infinity>() << std::endl;

    return;
}

