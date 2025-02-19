// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include "experiments/utils/fpca_extensive_testing.cpp"


#include "nlohmann/json.hpp"
using nlohmann::json;

#include "../../../fdaPDE-core/fdaPDE/utils/symbols.h"
#include "../../../fdaPDE-core/fdaPDE/linear_algebra.h"
using fdapde::core::RandomizedSVD;
using fdapde::core::IterationPolicy;


int main(int argc, char **argv){

    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);
    
    std::string rsvd_version = data["RunParams"].value("RSVDType","monolithic");
    double lambda = data["RunParams"].value("lambda",1e-2);

    // SVD solvers
    std::string svd_version = data["RunParams"].value("SVDType","exact");
    
    if(rsvd_version == "simple_monolithic"){
        fpca_test<fdapde::monolithic, SVDPolicy::JacobiSVD>(lambda);
    }
    if(rsvd_version == "fspai_monolithic"){
        fpca_test<fdapde::monolithic_fpsai,SVDPolicy::JacobiSVD>(lambda);
    }
    if(rsvd_version == "spchol_monolithic"){
        fpca_test<fdapde::monolithic_spchol,SVDPolicy::JacobiSVD>(lambda);
    }
    return 0;
}
