//
// Created by Marco Galliani on 02/12/24.
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

#include "experiments/cpp-utils/monoFPCA_solver.cpp"

template<typename ModelType>
void monoFPCA_calibration(ModelType &model, int rank,
                               Calibration calibration, DMatrix<double> lambda_grid, int n_folds,
                               double gcv_correction = 1.0,
                               int seed = fdapde::random_seed){
    //(1) Calibration
    DVector<double> optimal_lambda;
    const DMatrix<double> &X = model.X();

    DVector<double> cv_values(lambda_grid.size());

    auto mono_solver = MonoFPCASolver<ModelType>(model);

    const auto start{std::chrono::steady_clock::now()};
    switch (calibration){
        case Calibration::gcv: {
            // select \lambda minimizing the GCV index
            ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                mono_solver.compute(X,rank,lambda(0));
                return mono_solver.gcv(gcv_correction);
            });
            DVector<double> current_lambda = optimal_lambda = lambda_grid.row(0);
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
            //optimal_lambda = fdapde::core::Grid<Dynamic> {}.optimize(gcv, lambda_grid);
        } break;
        case Calibration::kcv:{
            // select \lambda minimizing the reconstruction error in cross-validation
            auto cv_score = [&](
                    const DVector<double>& lambda,
                    const fdapde::core::BinaryVector<Dynamic>& train_set,
                    const fdapde::core::BinaryVector<Dynamic>& test_set) -> double {
                //fitting on the training set
                mono_solver.compute(train_set.repeat(1,X.cols()).select(X),rank,lambda(0));
                //evaluate the error on the test set
                return mono_solver.reconstruction_error_simple(test_set.repeat(1,X.cols()).select(X));
            };
            auto KCV = fdapde::calibration::KCV{n_folds, seed};
            optimal_lambda = KCV.fit(model, lambda_grid, cv_score);
            cv_values = KCV.avg_scores();
        } break;
        case Calibration::kcv_cols: {
            // select \lambda minimizing the reconstruction error in cross-validation
            auto cv_score = [&](
                    const DVector<double>& lambda,
                    const fdapde::core::BinaryVector<Dynamic>& train_set,
                    const fdapde::core::BinaryVector<Dynamic>& test_set) -> double {

                DMatrix<double> Psi = model.Psi(); //cast to dense matrix for slicing
                Psi = Psi(train_set.which(true),Eigen::all);

                //for the columns deletion strategy we have to recompute the cholesky at every step
                Eigen::LLT<DMatrix<double>> chol(Psi.transpose()*Psi+model.P(lambda));
                DMatrix<double> invD = chol.matrixL().solve(DMatrix<double>::Identity(model.n_basis(),model.n_basis()));

                RSVD<DMatrix<double>> svd;
                svd.compute(X(Eigen::all,train_set.which(true))*Psi*invD.transpose(),rank);

                //evaluate error on the test set
                Psi = model.Psi(); //cast to dense matrix for slicing
                Psi = Psi(test_set.which(true),Eigen::all);
                DMatrix<double> S_m = Psi*invD.transpose();
                S_m = S_m*S_m.transpose();

                return (svd.matrixU().leftCols(rank).transpose()*X(Eigen::all,test_set.which(true))*(DMatrix<double>::Identity(S_m.rows(), S_m.cols()) - S_m)
                       ).squaredNorm() / (rank*test_set.size());
            };
            auto KCV = fdapde::calibration::KCV<fdapde::calibration::KCVPolicy::col_deletion>{n_folds, seed};
            optimal_lambda = KCV.fit(model, lambda_grid, cv_score);
            cv_values = KCV.avg_scores();
        } break;
        case Calibration::ocv: { //on the columns
            // select \lambda minimizing the GCV index
            ScalarField<Dynamic> ocv([&](const DVector<double>& lambda) -> double {
                mono_solver.compute(X,rank,lambda(0));
                return mono_solver.ocv();
            });
            DVector<double> current_lambda = lambda_grid.row(0);
            cv_values(0) = ocv(current_lambda);
            double optimal_value = cv_values(0);
            for(int i = 1; i < lambda_grid.rows(); ++i){
                current_lambda = lambda_grid.row(i);
                cv_values(i) = ocv(current_lambda);
                // update minimum if better optimum found
                if (cv_values(i) < optimal_value) {
                    optimal_value = cv_values(i);
                    optimal_lambda = current_lambda;
                }
            }
            //optimal_lambda = fdapde::core::Grid<Dynamic> {}.optimize(ocv, lambda_grid);
        } break;
        case Calibration::gcv_smooth: {
            // select \lambda minimizing the GCV index
            ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                mono_solver.compute(X,rank,lambda(0));
                return mono_solver.full_smoothing_gcv(gcv_correction);
            });
            DVector<double> current_lambda = lambda_grid.row(0);
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
            //optimal_lambda = fdapde::core::Grid<Dynamic> {}.optimize(gcv, lambda_grid);
        } break;
    }
    const auto end{std::chrono::steady_clock::now()};
    double calibration_time = (std::chrono::duration<double>{end - start}).count();

    //(2) Fit with optimal lambda
    mono_solver.compute(X,rank,optimal_lambda(0));

    //(3) Results
    DMatrix<double> scores = mono_solver.scores();
    DMatrix<double> loadings = mono_solver.loadings();
    DVector<double> loadings_norm(rank);
    for (int i = 0; i < rank; ++i) {
        loadings_norm[i] = std::sqrt(loadings.col(i).dot(model.R0() * loadings.col(i)));   // L^2 norm
        loadings.col(i) = loadings.col(i) / loadings_norm[i];
    }
    scores = scores.array().rowwise() * loadings_norm.transpose().array();

    //save results
    Eigen::saveMarket(model.Psi() * loadings,"results/estimated_PCs.mtx");
    Eigen::saveMarket(scores,"results/estimated_Scores.mtx");
    Eigen::saveMarket(cv_values,"results/CV_values.mtx");

    //test report: format [calibration_time,factorization_time,fit_time,lambda]
    std::ofstream test_report("results/test_report.csv");
    test_report << calibration_time << ",";
    test_report << mono_solver.factorization_time() << ",";
    test_report << mono_solver.fit_time() << ",";
    test_report << optimal_lambda(0) << std::endl;
    test_report.close();

    std::cout << "Lambda: " << optimal_lambda(0) << std::endl;
}

template<typename ModelType>
void fixed_monoFPCA_calibration(ModelType &model, int rank,
                          Calibration calibration, DMatrix<double> lambda_grid, int n_folds,
                          double gcv_correction = 1.0,
                          int seed = fdapde::random_seed){
    //(1) Calibration
    DVector<double> optimal_lambda;
    const DMatrix<double> &X = model.X();

    auto mono_solver = FixedSmoothingMonoFPCASolver<ModelType>(model,lambda_grid(0));

    const auto start{std::chrono::steady_clock::now()};
    switch (calibration) {
        case Calibration::gcv: {
            // select \lambda minimizing the GCV index
            ScalarField<Dynamic> gcv([&](const DVector<double>& lambda) -> double {
                mono_solver.compute(fdapde::core::BinaryMatrix<Dynamic,Dynamic>::Ones(X.rows(),X.cols()),rank,lambda(0));
                return mono_solver.gcv();
            });
            optimal_lambda = fdapde::core::Grid<Dynamic> {}.optimize(gcv, lambda_grid);
        } break;
        case Calibration::kcv:{
            // select \lambda minimizing the reconstruction error in cross-validation
            auto cv_score = [&](
                    const DVector<double>& lambda,
                    const fdapde::core::BinaryVector<Dynamic>& train_set,
                    const fdapde::core::BinaryVector<Dynamic>& test_set) -> double {
                //fitting on the training set
                mono_solver.compute(train_set.repeat(1,X.cols()),rank,lambda(0));
                //evaluate the error on the test set
                return mono_solver.reconstruction_error(test_set.repeat(1,X.cols()));
            };
            optimal_lambda =
                    fdapde::calibration::KCV{n_folds, seed}.fit(model, lambda_grid, cv_score);
        } break;
    }
    const auto end{std::chrono::steady_clock::now()};
    double calibration_time = (std::chrono::duration<double>{end - start}).count();

    //(2) Fit with optimal lambda
    mono_solver.compute(fdapde::core::BinaryMatrix<Dynamic,Dynamic>::Ones(X.rows(),X.cols()),rank,optimal_lambda(0));

    //(3) Results
    DMatrix<double> scores = mono_solver.scores();
    DMatrix<double> loadings = mono_solver.loadings();
    DVector<double> loadings_norm(rank);
    for (int i = 0; i < rank; ++i) {
        loadings_norm[i] = std::sqrt(loadings.col(i).dot(model.R0() * loadings.col(i)));   // L^2 norm
        loadings.col(i) = loadings.col(i) / loadings_norm[i];
    }
    scores = scores.array().rowwise() * loadings_norm.transpose().array();

    //save results
    Eigen::saveMarket(model.Psi() * loadings,"results/estimated_PCs.mtx");
    Eigen::saveMarket(scores,"results/estimated_Scores.mtx");

    //test report: format [calibration_time,factorization_time,fit_time,lambda]
    std::ofstream test_report("results/test_report.csv");
    test_report << calibration_time << ",";
    test_report << mono_solver.factorization_time() << ",";
    test_report << mono_solver.fit_time() << ",";
    test_report << optimal_lambda(0) << std::endl;
    test_report.close();

    std::cout << "Lambda: " << optimal_lambda(0) << std::endl;
}

enum CalibrationProcedure{
        direct,
        generalized
    };

void monoFPCA_calibration_test(DVector<double> lambda_grid, Calibration cal, int n_pc=3, int n_folds=10, double gcv_correction=1.0,
                               CalibrationProcedure cal_procedure=CalibrationProcedure::generalized){
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    // import locations
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test/locs.csv");
    //regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);

    //dummy rsvd just to construct the FPCA object
    RegularizedSVD<fdapde::monolithic,RSVD<DMatrix<double>>> dummy_rsvd;

    //define the model
    FPCA<SpaceOnly> model(pde, Sampling::pointwise, dummy_rsvd);
    model.set_spatial_locations(locs);
    //set model's data
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test/datamatrix_centred.csv");
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    model.set_npc(n_pc);

    model.init();
    //call to the calibrated model
    switch (cal_procedure) {
        case CalibrationProcedure::generalized: {
            monoFPCA_calibration<decltype(model)>(model, n_pc, cal, lambda_grid, n_folds, gcv_correction);
        } break;
        case CalibrationProcedure::direct: {
            fixed_monoFPCA_calibration<decltype(model)>(model, n_pc, cal, lambda_grid, n_folds, gcv_correction);
        } break;
    }
    return;
}

