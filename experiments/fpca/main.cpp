// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include "experiments/cpp-utils/fpca_extensive_testing.cpp"


#include "nlohmann/json.hpp"
using nlohmann::json;

#include "../../../fdaPDE-core/fdaPDE/utils/symbols.h"
#include "../../../fdaPDE-core/fdaPDE/linear_algebra.h"
using fdapde::core::RSVD;

using fdapde::monolithic;
using fdapde::sequential;

int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);

    std::string rsvd_version = data["RunParams"].value("RSVDType","monolithic");
    std::string svd_version = data["RunParams"].value("SVDType","exact");
    double lambda = data["RunParams"].value("lambda",1e-5);


    if(rsvd_version == "monolithic"){
        if(svd_version == "exact") fpca_test<monolithic,Eigen::JacobiSVD<DMatrix<double>>>(lambda);
        if(svd_version == "randomized") fpca_test<monolithic,RSVD<DMatrix<double>>>(lambda);

    }
    if(rsvd_version == "sequential"){
        if(svd_version == "exact") fpca_test<sequential,Eigen::JacobiSVD<DMatrix<double>>>(lambda);
        if(svd_version == "randomized") fpca_test<sequential,RSVD<DMatrix<double>>>(lambda);
    }
    return 0;
}
