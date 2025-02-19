// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <map>

#include "experiments/cpp-utils/fpca_extensive_testing.cpp"

#include "nlohmann/json.hpp"
using nlohmann::json;

#include "../../../fdaPDE-core/fdaPDE/utils/symbols.h"
#include "../../../fdaPDE-core/fdaPDE/linear_algebra.h"

int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);
    
    std::string rsvd_version = data["RunParams"].value("RSVDType","monolithic");
    std::string svd_version = data["RunParams"].value("SVDType","exact");

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



    if (rsvd_version == "monolithic") {
        if(svd_version=="exact") scores_computation<fdapde::monolithic,Eigen::JacobiSVD<DMatrix<double>>>(fixed_lambda, n_pc);
        else if (svd_version=="randomized") scores_computation<fdapde::monolithic>(fixed_lambda, n_pc);
    } else {
        if(svd_version=="exact") scores_computation<fdapde::sequential,Eigen::JacobiSVD<DMatrix<double>>>(fixed_lambda, n_pc);
        else if (svd_version=="randomized") scores_computation<fdapde::sequential>(fixed_lambda, n_pc);
    }

    return 0;
}
