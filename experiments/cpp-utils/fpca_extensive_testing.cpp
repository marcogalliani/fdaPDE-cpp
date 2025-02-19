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
#include "fdaPDE/calibration/symbols.h"
using fdapde::calibration::Calibration;

#include "test/src/utils/constants.h"
#include "test/src/utils/mesh_loader.h"
#include "test/src/utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

template<typename RSVDPolicy_,typename SVDPolicy_=RSVD<DMatrix<double>>>
void fpca_test(double lambda = 1e-2, int n_pc=3){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define the regularized svd solver
    RegularizedSVD<RSVDPolicy_,SVDPolicy_> rsvd;
    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_spatial_locations(locs);
    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    //penalization
    model.set_lambda_D(lambda);
    model.set_npc(n_pc);
    //solve FPCA problem
    const auto start{std::chrono::steady_clock::now()};
    model.init();
    model.solve();
    const auto end{std::chrono::steady_clock::now()};
    //Results
    Eigen::saveMarket(model.Psi() * model.loadings(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(model.scores(),"results/estimated_Scores.mtx");
    std::ofstream exe_times_file("results/execution_times.csv");
    exe_times_file << (std::chrono::duration<double>{end - start}).count() << std::endl;
    exe_times_file.close();

    return;
}

template<typename RSVDPolicy_,typename SVDPolicy_=RSVD<DMatrix<double>>>
void fpca_calibration_test(DVector<double> lambda_grid, Calibration cal, int n_pc=3, int n_folds=10){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define the regularized svd solver
    RegularizedSVD<RSVDPolicy_,SVDPolicy_> rsvd(cal);
    rsvd.set_lambda(lambda_grid);
    if(cal==Calibration::kcv_cols){
        rsvd.set_nfolds(n_folds);
    }
    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_spatial_locations(locs);
    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    model.set_npc(n_pc);

    //solve FPCA problem
    const auto start{std::chrono::steady_clock::now()};
    model.init();
    model.solve();
    const auto end{std::chrono::steady_clock::now()};

    Eigen::saveMarket(model.Psi()*model.loadings(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(model.scores(),"results/estimated_Scores.mtx");
}

template<typename RSVDPolicy_,typename SVDPolicy_=RSVD<DMatrix<double>>>
void scores_computation(double lambda, int n_pc=3){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    // define the regularized svd solver
    RegularizedSVD<RSVDPolicy_,SVDPolicy_> rsvd;

    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    Eigen::ArrayXi indices = Eigen::ArrayXi::LinSpaced(y.rows(),0,y.rows()-1);
    std::mt19937 rng(7050);
    std::shuffle(indices.data(),indices.data()+y.rows(),rng);
    int n_test = y.rows()/10;

    DMatrix<double> X_test_true = y(indices.head(n_test),Eigen::all);
    y = y + fdapde::internals::GaussianMatrix(y.rows(),y.cols(),7050,1.0);

    DMatrix<double> X_test = y(indices.head(n_test),Eigen::all);
    DMatrix<double> X_train = y(indices.tail(y.rows()-n_test),Eigen::all);

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, X_train);
    model.set_data(df);
    model.set_npc(n_pc);

    //solve FPCA problem
    const auto start{std::chrono::steady_clock::now()};
    model.init();
    model.solve();
    const auto end{std::chrono::steady_clock::now()};

    //->loadings normalization
    DMatrix<double> loadings = model.loadings();
    DVector<double> loadings_norm = model.loadings_norm();
    loadings = loadings.array().rowwise() * loadings_norm.transpose().array();

    DMatrix<double> true_scores = model.scores().array().rowwise()/loadings_norm.transpose().array();

    loadings_norm = (loadings.transpose()*(model.Psi().transpose()*model.Psi()+model.P())*loadings).diagonal();
    loadings_norm = loadings_norm.cwiseSqrt();
    loadings = loadings.array().rowwise() / loadings_norm.transpose().array();
    true_scores = true_scores.array().rowwise()*loadings_norm.transpose().array();

    //->scores computation
    Eigen::SimplicialLLT<SpMatrix<double>> chol; //cholesky of Psi^T*Psi
    chol.compute(model.Psi().transpose()*model.Psi());

    DMatrix<double> est_scores_2 = model.X()*model.Psi()*loadings;
    DMatrix<double> est_scores_3 = model.X()*model.Psi()*loadings*(loadings.transpose()*model.Psi().transpose()*model.Psi()*loadings).inverse();

    //orthogonality
    std::cout << "Orthogonality of the PCs w.r.t. the smoothing matrix:" << std::endl;
    std::cout << (loadings.transpose()*(model.Psi().transpose()*model.Psi()+model.P())*loadings) << std::endl;

    std::cout << "Orthogonality of the scores:" << std::endl;
    std::cout << model.scores().transpose()*model.scores() << std::endl;

    DMatrix<double> E; //residual matrix

    std::cout << "Difference between the scores:" << std::endl;

    std::cout << "-> method 2" << std::endl;
    std::cout << (true_scores-est_scores_2).colwise().norm() << std::endl;
    E = model.X() - est_scores_2*loadings.transpose()*model.Psi().transpose();
    std::cout << E.norm()/std::sqrt(model.X().rows()*model.X().cols()) << std::endl;
    std::cout << (E*model.Psi()*loadings).norm() << std::endl;
    std::cout << "-> method 3" << std::endl;
    std::cout << (true_scores-est_scores_3).colwise().norm() << std::endl;
    E = model.X() - est_scores_3*loadings.transpose()*model.Psi().transpose();
    std::cout << E.norm()/std::sqrt(model.X().rows()*model.X().cols()) << std::endl;
    std::cout << (E*model.Psi()*loadings).norm() << std::endl;
}


