// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <map>
#include "experiments/cpp-utils/mono-calibration.cpp"


#include "nlohmann/json.hpp"
using nlohmann::json;

#include "../../../fdaPDE-core/fdaPDE/utils/symbols.h"
#include "../../../fdaPDE-core/fdaPDE/linear_algebra.h"

int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);

    Eigen::setNbThreads(data["RunParams"].value("n_cores",1));
    std::cout << "Number of cores: " << Eigen::nbThreads() << std::endl;

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
                   {"full_smooth_gcv", Calibration::gcv_smooth}});
    std::string cal_strategy = data["calibration"].value("strategy", "off");

    //Calibration procedure
    std::map<std::string,CalibrationProcedure> procedure_map;
    procedure_map.insert({{"direct", CalibrationProcedure::direct},
                    {"generalized", CalibrationProcedure::generalized}});
    std::string cal_procedure = data["calibration"].value("procedure", "generalized");

    //gcv correction
    double gcv_correction = data["calibration"].value("gcv_correction",1.0);

    //number of components
    int n_pc = data["RunParams"].value("n_pc",4);
    int n_folds = data["calibration"].value("n_folds",10);

    //test
    monoFPCA_calibration_test(lambda_grid, cal_map[cal_strategy],n_pc,n_folds,gcv_correction,procedure_map[cal_procedure]);

    return 0;
}
