// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <map>
#include "experiments/cpp-utils/fpca_extensive_testing.cpp"
#include "experiments/cpp-utils/npc_calibration.cpp"


#include "nlohmann/json.hpp"
using nlohmann::json;

#include "../../../fdaPDE-core/fdaPDE/utils/symbols.h"
#include "../../../fdaPDE-core/fdaPDE/linear_algebra.h"

int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);

    // lambda grid
    int grid_dim = data["lambda_grid"].value("dim",10);
    double left = data["lambda_grid"].value("left",-10);
    double right = data["lambda_grid"].value("right",-3);

    DMatrix<double> lambda_grid = DMatrix<double>::Constant(grid_dim,1,10.0);
    lambda_grid.col(0) = lambda_grid.col(0).array().pow(DVector<double>::LinSpaced(grid_dim,left,right).array());

    // n_pc grid
    int npc_left = data["npc_grid"].value("left",-10);
    int npc_right = data["npc_grid"].value("right",-3);

    std::vector<int> npc_grid(npc_right-npc_left + 1);
    for(int i=0; i < npc_grid.size(); i++){
        npc_grid[i] = npc_left + i;
    }

    //Calibration strategy
    std::map<std::string,Calibration> cal_map;
    cal_map.insert({
        {"off", Calibration::off},
        {"kcv", Calibration::kcv},
        {"gcv", Calibration::gcv},
        {"new_gcv", Calibration::gcv_smooth}});
    std::string cal_strategy = data["Calibration"].value("strategy", "kcv");
    int n_folds = data["Calibration"].value("n_folds", 10);

    //if calibration off
    double fixed_lambda = data["RunParams"].value("fixed_lambda",0.00001);
    int fixed_npc = data["RunParams"].value("fixed_npc",3);


    if(cal_strategy=="off"){
        fpca_test<fdapde::monolithic>(fixed_lambda,fixed_npc);
    }else{
        npc_calibration_test(lambda_grid, npc_grid,
                             cal_map[cal_strategy], n_folds);
    }
    return 0;
}
