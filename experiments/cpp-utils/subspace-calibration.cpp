//
// Created by Marco Galliani on 17/02/25.
//

#include "fdaPDE/core.h"
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

#include "fdaPDE/models/functional/fpca.h"
using fdapde::models::FPCA;
using fdapde::models::RegularizedSVD;
using fdapde::models::Sampling;

#include "test/src/utils/constants.h"
#include "test/src/utils/mesh_loader.h"
#include "test/src/utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

//Eigen
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

void subFPCA_calibration_test(DMatrix<double> lambda_grid, Calibration calibration, int n_folds=10, int n_pcs=3, int seed=fdapde::random_seed){
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

    model.init();
    RSVD<DMatrix<double>> svd;

    //(1) Initialise SubspaceIterations solver
    fdapde::models::FixedLambdaSubspaceIteration<decltype(model)> subspace_solver(model, 1e-6, 150, 1412);

    //(2) Select the optimal \lambda
    DVector<double> optimal_lambda;
    DVector<double> cv_values(lambda_grid.size());
    switch (calibration) {
        case Calibration::gcv:{
            svd.compute(y,3);
            ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                subspace_solver.compute(y, 3, lambda, svd.matrixV().leftCols(3));
                return subspace_solver.gcv(y);   // return GCV index at convergence
            });
            DVector<double> current_lambda = optimal_lambda =  lambda_grid.row(0);
            cv_values(0) =  gcv(current_lambda);
            double optimal_value = cv_values(0);
            for(int i = 1; i < lambda_grid.rows(); ++i) {
                current_lambda = lambda_grid.row(i);
                cv_values(i) = gcv(current_lambda);
                // update minimum if better optimum found
                if (cv_values(i) < optimal_value) {
                    optimal_value = cv_values(i);
                    optimal_lambda = current_lambda;
                }
            }
        } break;
        case Calibration::kcv:{
            auto cv_score = [&](
                    const DVector<double>& lambda,
                    const fdapde::core::BinaryVector<Dynamic>& train_set,
                    const fdapde::core::BinaryVector<Dynamic>& test_set) -> double {

                //fit the model on the training set
                svd.compute(DMatrix<double>(train_set.repeat(1,y.cols()).select(y)),3);
                subspace_solver.compute(train_set.repeat(1,y.cols()).select(y), 3, lambda, svd.matrixV().leftCols(3));

                //compute the representation of the fPCA as a linear smoother
                DVector<double> loadings_S_m_norms = (subspace_solver.F().transpose()*(model.Psi().transpose()*model.Psi()+model.P(lambda))*subspace_solver.F()).diagonal();
                DMatrix<double> S_m = subspace_solver.Fn()*(loadings_S_m_norms.array().inverse()).matrix().asDiagonal()*subspace_solver.Fn().transpose();

                //evaluate the error on the test set
                double cv_score = (test_set.repeat(1,y.cols()).select(y)*(DMatrix<double>::Identity(y.cols(),y.cols())-S_m)).norm()/std::sqrt(test_set.size()*y.cols());
                return cv_score;
            };
            auto KCV = fdapde::calibration::KCV{n_folds, seed};
            optimal_lambda = KCV.fit(model, lambda_grid, cv_score);
            cv_values = KCV.avg_scores();
        } break;
    }

    //(3) Run with the optimal lambda
    svd.compute(y,3);
    subspace_solver.compute(y, 3, optimal_lambda, svd.matrixV().leftCols(3));

    Eigen::saveMarket(subspace_solver.Fn(),"results/estimated_PCs.mtx");
    Eigen::saveMarket(subspace_solver.S(),"results/estimated_Scores.mtx");
    Eigen::saveMarket(cv_values,"results/CV_values.mtx");

    std::cout << "Lambda: " << optimal_lambda(0) << std::endl;
}