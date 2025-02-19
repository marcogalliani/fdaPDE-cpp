// include eigen now to avoid possible linking errors
#include <Eigen/Dense>

#include "experiments/cpp-utils/fpca_extensive_testing.cpp"
#include "experiments/cpp-utils/fpca_subspace.cpp"


#include "nlohmann/json.hpp"
using nlohmann::json;

int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);
    std::string rsvd_version = data["RunParams"].value("RSVDType","monolithic");

    //std::vector<double> lambda_PCs = data["RunParams"]["lambda_PCs"].get<std::vector<double>>();
    //lambda grid
    double lower_bound = data["Calibration"].value("LB",-10);
    double upper_bound = data["Calibration"].value("UB",-1);
    int grid_dim = data["Calibration"].value("grid_dim",30);
    DMatrix<double> lambda_grid = DMatrix<double>::Constant(grid_dim,1,10.0);
    lambda_grid.col(0) = lambda_grid.col(0).array().pow(DVector<double>::LinSpaced(grid_dim,lower_bound,upper_bound).array());

    if(rsvd_version == "sequential") fpca_calibration_test<fdapde::sequential>(lambda_grid,Calibration::gcv,3,10);
    else if(rsvd_version == "subspace") fpca_subspace_fixed_calibration(lambda_grid,3);
    else if(rsvd_version == "monolithic") fpca_calibration_test<fdapde::monolithic>(lambda_grid,Calibration::gcv,3,10);

    return 0;
}
