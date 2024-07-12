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
void fpca_test(SVDType svd){
    // define domain
    MeshLoader<Mesh2D> domain("unit_square_rsvd_test");

    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test_rsvd/locs.csv");
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test_rsvd/datamatrix_centred-1.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, RegularizedSVD<SolutionPolicy_,SVDType>(svd));
    model.set_spatial_locations(locs);
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
    DVector<double> f1 = read_csv<double>("../data/models/fpca/2D_test_rsvd/f1.csv");
    DVector<double> f2 = read_csv<double>("../data/models/fpca/2D_test_rsvd/f2.csv");
    DVector<double> f3 = read_csv<double>("../data/models/fpca/2D_test_rsvd/f3.csv");

    DVector<double> est_f1 = (model.Psi() * model.loadings()).col(0);
    DVector<double> est_f2 = (model.Psi() * model.loadings()).col(1);
    DVector<double> est_f3 = (model.Psi() * model.loadings()).col(2);

    //1st autofunction
    std::cout << "First PC" << std::endl;
    std::cout << (f1 - est_f1).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (f1 + est_f1).lpNorm<Eigen::Infinity>() << std::endl;
    //2nd autofunction
    std::cout << "Second PC" << std::endl;
    std::cout << (f2 - est_f2).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (f2 + est_f2).lpNorm<Eigen::Infinity>() << std::endl;
    //3rd autofunction
    std::cout << "Third PC" << std::endl;
    std::cout << (f3 - est_f3).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (f3 + est_f3).lpNorm<Eigen::Infinity>() << std::endl;

    //-> Scores
    DVector<double> scores_1 = read_csv<double>("../data/models/fpca/2D_test_rsvd/score1-1.csv");
    DVector<double> scores_2 = read_csv<double>("../data/models/fpca/2D_test_rsvd/score2-1.csv");
    DVector<double> scores_3 = read_csv<double>("../data/models/fpca/2D_test_rsvd/score3-1.csv");

    DVector<double> est_scores_1 = model.scores().col(0);
    DVector<double> est_scores_2 = model.scores().col(1);
    DVector<double> est_scores_3 = model.scores().col(2);

    //1st PC scores
    std::cout << "First PC scores" << std::endl;
    std::cout << (scores_1 - est_scores_1).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (scores_1 + est_scores_1).lpNorm<Eigen::Infinity>() << std::endl;
    //2nd scores
    std::cout << "Second PC scores" << std::endl;
    std::cout << (scores_2 - est_scores_2).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (scores_2 - est_scores_2).lpNorm<Eigen::Infinity>() << std::endl;
    //3rd scores
    std::cout << "Third PC scores" << std::endl;
    std::cout << (scores_3 - est_scores_3).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << (scores_3 - est_scores_3).lpNorm<Eigen::Infinity>() << std::endl;

    return;
}
