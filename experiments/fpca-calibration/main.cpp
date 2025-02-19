// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <map>

#include "experiments/cpp-utils/fpca_extensive_testing.cpp"
#include "experiments/cpp-utils/mono-calibration.cpp"
#include "experiments/cpp-utils/seq-calibration.cpp"
#include "experiments/cpp-utils/subspace-calibration.cpp"
#include "experiments/cpp-utils/fpca_subspace.cpp"


#include "nlohmann/json.hpp"
using nlohmann::json;

#include "../../../fdaPDE-core/fdaPDE/utils/symbols.h"
#include "../../../fdaPDE-core/fdaPDE/linear_algebra.h"

int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);
    
    std::string rsvd_version = data["RunParams"].value("RSVDType","monolithic");

    // grid of smoothing parameters
    int grid_dim = data["calibration"].value("grid_dim",10);
    double left = data["calibration"].value("left",-10);
    double right = data["calibration"].value("right",-3);

    DMatrix<double> lambda_grid = DMatrix<double>::Constant(grid_dim,1,10.0);
    lambda_grid.col(0) = lambda_grid.col(0).array().pow(DVector<double>::LinSpaced(grid_dim,left,right).array());

    //Calibration strategy
    std::map<std::string,Calibration> cal_map;
    cal_map.insert({{"off", Calibration::off},
                   {"kcv", Calibration::kcv},
                   {"gcv", Calibration::gcv},
                   {"kcv_cols", Calibration::kcv_cols},
                   {"ocv", Calibration::ocv},
                   {"full_smooth_gcv", Calibration::gcv_smooth},
                   {"kcv_projection", Calibration::kcv_projection},
                   {"kcv_svd_extraction", Calibration::kcv_svd_extraction}
    });
    std::string cal_strategy = data["calibration"].value("strategy", "off");

    //number of components
    int n_pc = data["RunParams"].value("n_pc",4);
    int n_folds = data["calibration"].value("n_folds",10);

    double fixed_lambda = data["calibration"].value("fixed",0.00001);
    if(cal_map[cal_strategy] == Calibration::off){
        if (rsvd_version == "monolithic") fpca_test<fdapde::monolithic>(fixed_lambda);
        else if (rsvd_version == "sequential") fpca_test<fdapde::sequential>(fixed_lambda);
        else if (rsvd_version == "subspace") fpca_subspace_fixed(fixed_lambda, 100);

    }else{
        if (rsvd_version == "monolithic") monoFPCA_calibration_test(lambda_grid, cal_map[cal_strategy],n_pc,n_folds);
        else if (rsvd_version=="sequential") seqFPCA_calibration_test(lambda_grid, cal_map[cal_strategy],n_pc,n_folds);
        else if (rsvd_version=="subspace") subFPCA_calibration_test(lambda_grid, cal_map[cal_strategy],n_folds);
    }
    return 0;
}
