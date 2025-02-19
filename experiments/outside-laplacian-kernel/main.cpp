// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <map>

#include "experiments/cpp-utils/fpca_extensive_testing.cpp"
#include "experiments/cpp-utils/mono-calibration.cpp"


#include "nlohmann/json.hpp"
using nlohmann::json;

#include "../../../fdaPDE-core/fdaPDE/utils/symbols.h"
#include "../../../fdaPDE-core/fdaPDE/linear_algebra.h"

int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);

    //RSVD solver
    std::string rsvd_policy = data["RunParams"].value("RSVDType","monolithic");
    //SVD solver
    std::string svd_type = data["RunParams"].value("SVDType","exact");

    Eigen::setNbThreads(data["RunParams"].value("n_cores",1));
    std::cout << "Number of cores: " << Eigen::nbThreads() << std::endl;

    // grid of smoothing parameters
    double fixed_lambda = data["RunParams"].value("lambda",1e-5);
    //number of components
    int n_pc = data["RunParams"].value("n_pc",4);

    if (svd_type == "randomized"){
        if(rsvd_policy == "sequential"){
            fpca_test<fdapde::sequential,RSVD<DMatrix<double>>>(fixed_lambda);
        }
        if(rsvd_policy == "monolithic"){
            fpca_test<fdapde::monolithic,RSVD<DMatrix<double>>>(fixed_lambda);
        }
    } else if(svd_type == "exact"){
        if(rsvd_policy == "sequential"){
            fpca_test<fdapde::sequential,Eigen::JacobiSVD<DMatrix<double>>>(fixed_lambda);
        }
        if(rsvd_policy == "monolithic"){
            fpca_test<fdapde::monolithic,Eigen::JacobiSVD<DMatrix<double>>>(fixed_lambda);
        }
    }
    return 0;
}
