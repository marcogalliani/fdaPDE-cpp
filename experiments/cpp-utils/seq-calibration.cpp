//
// Created by Marco Galliani on 13/12/24.
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
using fdapde::models::PowerIteration;
using fdapde::models::StochasticEDF;

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



template<typename ModelType>
void seqFPCA_calibration(ModelType &model, int rank,
                          Calibration calibration, DMatrix<double> lambda_grid, int n_folds=10,
                          double gcv_correction = 1.0,
                          int seed = fdapde::random_seed){
    //(1) Calibration
    DVector<double> optimal_lambda;
    const DMatrix<double> X = model.X();
    BlockFrame<double, int> df;

    DVector<double> cv_values(lambda_grid.size());

    const auto start{std::chrono::steady_clock::now()};
    switch (calibration) {
        case Calibration::gcv:{
            RSVD<DMatrix<double>> svd;
            svd.compute(X, rank);
            // select \lambda minimizing the GCV index
            ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                model.set_lambda(lambda);
                model.init();
                PowerIteration<ModelType> solver(model,1e-6,20,seed);
                solver.init();

                DMatrix<double> S(X.rows(),rank);
                DMatrix<double> F(model.n_basis(),rank);
                DMatrix<double> X_d = X;
                //->sequential estimation of the components
                for (int index = 0; index < rank; index++) {
                    //fit on the imputed data
                    solver.compute(X_d, lambda, svd.matrixV().col(index));
                    //deflation
                    X_d -= solver.s() * solver.fn().transpose() * solver.f_norm();
                    //normalization
                    F.col(index) = solver.f()*solver.f_norm();
                    S.col(index) = solver.s() ;
                }
                return (S.transpose()*X-(model.Psi()*F).transpose()).squaredNorm()*X.cols()/std::pow(X.cols()-gcv_correction*solver.edfs(),2);
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
            //optimal_lambda = fdapde::core::Grid<Dynamic>{}.optimize(gcv, lambda_grid);
        } break;
        case Calibration::kcv:{
            // select \lambda minimizing the reconstruction error in cross-validation
            auto cv_score = [&](
                    const DVector<double>& lambda,
                    const fdapde::core::BinaryVector<Dynamic>& train_set,
                    const fdapde::core::BinaryVector<Dynamic>& test_set) -> double {

                //fit the model on the training set
                model.set_lambda(lambda);
                df.insert(OBSERVATIONS_BLK, DMatrix<double>(train_set.repeat(1,X.cols()).select(X)));
                model.set_data(df);

                model.init();
                model.solve();

                DMatrix<double> loadings = model.loadings();
                DVector<double> loadings_norm = (loadings.transpose()*(model.Psi().transpose()*model.Psi()+model.P(lambda))*loadings).diagonal();
                loadings_norm = loadings_norm.cwiseSqrt();

                loadings = loadings.array().rowwise() / loadings_norm.transpose().array();
                DMatrix<double> test_scores = test_set.repeat(1,X.cols()).select(X)*model.Psi()*loadings;

                //evaluate the error on the test set
                double cv_score = (test_set.repeat(1,X.cols()).select(X)-test_scores*loadings.transpose()*model.Psi().transpose()).norm()/std::sqrt(test_set.size()*X.cols());
                return cv_score;
            };
            auto KCV = fdapde::calibration::KCV{n_folds, seed};
            optimal_lambda = KCV.fit(model, lambda_grid, cv_score);
            cv_values = KCV.avg_scores();
        } break;
    }
    const auto end{std::chrono::steady_clock::now()};
    double calibration_time = (std::chrono::duration<double>{end - start}).count();

    //(2) Fit with optimal lambda
    model.set_lambda(optimal_lambda);
    df.insert(OBSERVATIONS_BLK, X);
    model.set_data(df);

    model.init();
    model.solve();

    //(3) Results
    Eigen::saveMarket(model.Psi() * model.loadings(), "results/estimated_PCs.mtx");
    Eigen::saveMarket(model.scores(), "results/estimated_Scores.mtx");
    Eigen::saveMarket(cv_values,"results/CV_values.mtx");

    //test report: format [calibration_time,factorization_time,fit_time,lambda]
    std::ofstream test_report("results/test_report.csv");
    test_report << calibration_time << ",";
    test_report << optimal_lambda(0) << std::endl;
    test_report.close();

    std::cout << "Lambda: " << optimal_lambda(0) << std::endl;
}

void seqFPCA_calibration_test(DVector<double> lambda_grid, Calibration cal, int n_pc=3, int n_folds=10, double gcv_correction=1.0){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);

    RegularizedSVD<fdapde::sequential,RSVD<DMatrix<double>>> rsvd;

    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
    model.set_spatial_locations(locs);

    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    model.set_npc(n_pc);

    model.init();
    //call to the calibrated model
    seqFPCA_calibration(model,n_pc,cal,lambda_grid,n_folds);
    return;
}

