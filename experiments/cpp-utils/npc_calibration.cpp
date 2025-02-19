//
// Created by Marco Galliani on 10/01/25.
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
void npc_calibration(ModelType &model,
                     std::vector<int> npc_grid,
                     DMatrix<double> lambda_grid,
                     Calibration calibration,
                     int n_folds,
                     double gcv_correction = 1.0,
                     int seed = fdapde::random_seed){

    //(1) Calibration
    DVector<double> optimal_lambda;
    int optimal_rank;
    const DMatrix<double> &X = model.X();

    DMatrix<double> cv_values(lambda_grid.rows(),npc_grid.size());
    auto mono_solver = MonoFPCASolver<ModelType>(model);

    const auto start{std::chrono::steady_clock::now()};
    switch (calibration){
        case Calibration::gcv: {
            // select \lambda minimizing the GCV index
            auto gcv = [&](const DVector<double>& lambda, int rank){
                mono_solver.compute(X,rank,lambda(0));
                return mono_solver.gcv(gcv_correction);
            };
            DVector<double> current_lambda = optimal_lambda = lambda_grid.row(0);
            int current_rank = optimal_rank = npc_grid[0];
            cv_values(0,0) =  gcv(current_lambda,current_rank);
            double optimal_value = cv_values(0,0);
            for(int i = 0; i < lambda_grid.rows(); ++i) {
                for(int j=0; j < npc_grid.size(); j++){
                    current_lambda = lambda_grid.row(i);
                    current_rank = npc_grid[j];
                    cv_values(i,j) = gcv(current_lambda,current_rank);
                    // update minimum if better optimum found
                    if (cv_values(i,j) < optimal_value){
                        optimal_value = cv_values(i,j);
                        optimal_lambda = current_lambda;
                        optimal_rank = current_rank;
                    }
                }
            }
            //optimal_lambda = fdapde::core::Grid<Dynamic> {}.optimize(gcv, lambda_grid);
        } break;
        case Calibration::gcv_smooth: {
            // select \lambda minimizing the GCV index
            auto gcv = [&](const DVector<double>& lambda, int rank){
                mono_solver.compute(X,rank,lambda(0));
                return mono_solver.full_smoothing_gcv(gcv_correction);
            };
            DVector<double> current_lambda = optimal_lambda = lambda_grid.row(0);
            int current_rank = optimal_rank = npc_grid[0];
            cv_values(0,0) =  gcv(current_lambda,current_rank);
            double optimal_value = cv_values(0,0);
            for(int i = 0; i < lambda_grid.rows(); ++i) {
                for(int j=0; j < npc_grid.size(); j++){
                    current_lambda = lambda_grid.row(i);
                    current_rank = npc_grid[j];
                    cv_values(i,j) = gcv(current_lambda,current_rank);
                    // update minimum if better optimum found
                    if (cv_values(i,j) < optimal_value){
                        optimal_value = cv_values(i,j);
                        optimal_lambda = current_lambda;
                        optimal_rank = current_rank;
                    }
                }
            }
            //optimal_lambda = fdapde::core::Grid<Dynamic> {}.optimize(gcv, lambda_grid);
        } break;
        case Calibration::kcv:{
            // select \lambda minimizing the reconstruction error in cross-validation
            //hyperparams contains [lambda, npc]
            auto cv_score = [&](
                    const DVector<double>& hyperparams,
                    const fdapde::core::BinaryVector<Dynamic>& train_set,
                    const fdapde::core::BinaryVector<Dynamic>& test_set) -> double {
                //fitting on the training set
                mono_solver.compute(train_set.repeat(1,X.cols()).select(X),hyperparams(1),hyperparams(0));
                //evaluate the error on the test set
                return mono_solver.reconstruction_error(test_set.repeat(1,X.cols()).select(X));
            };
            auto KCV = fdapde::calibration::KCV{n_folds, seed};
            DMatrix<double> hyperparams_grid(lambda_grid.rows()*npc_grid.size(),2);
            for(int i = 0; i < lambda_grid.rows(); ++i) {
                for(int j=0; j < npc_grid.size(); j++){
                    int index = i*npc_grid.size() + j;
                    hyperparams_grid(index,0) = lambda_grid(i,0);
                    hyperparams_grid(index,1) = npc_grid[j];
                }
            }
            DVector<double> optimal_hyperparams = KCV.fit(model, hyperparams_grid, cv_score);
            optimal_lambda = optimal_hyperparams(0)*DVector<double>::Ones(1);
            optimal_rank = optimal_hyperparams(1);
            cv_values = KCV.avg_scores().reshaped<Eigen::RowMajor>(lambda_grid.rows(),npc_grid.size());
        } break;
    }
    const auto end{std::chrono::steady_clock::now()};
    double calibration_time = (std::chrono::duration<double>{end - start}).count();

    //(2) Fit with optimal lambda
    mono_solver.compute(X,optimal_rank,optimal_lambda(0));

    //(3) Results
    DMatrix<double> scores = mono_solver.scores();
    DMatrix<double> loadings = mono_solver.loadings();
    DVector<double> loadings_norm(optimal_rank);
    for (int i = 0; i < optimal_rank; ++i) {
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
    std::cout << "Rank: " << optimal_rank << std::endl;
}

void npc_calibration_test(DMatrix<double> lambda_grid,
                          std::vector<int> npc_grid,
                          Calibration cal,
                          int n_folds=10,
                          double gcv_correction=1.0){
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

    model.init();
    //call to the calibrated model
    npc_calibration(model,
                    npc_grid, lambda_grid,
                    cal, n_folds, gcv_correction);

    return;
}


