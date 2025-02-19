// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include "experiments/cpp-utils/missing_fpca_test.cpp"

#include "nlohmann/json.hpp"
using nlohmann::json;


int main(int argc, char **argv){
    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);
    
    std::string rsvd_policy = data["RunParams"].value("RSVDType","monolithic");

    double lambda_D = data["RunParams"].value("lambda_D",1e-5);
    double lambda_T = data["RunParams"].value("lambda_T",1e-5);

    int t_steps = data["RunParams"].value("t_steps",15);

    if(rsvd_policy == "sequential"){
        st_fpca_miss_test<fdapde::sequential>(lambda_D,lambda_T,t_steps);
    }
    if(rsvd_policy == "monolithic"){
        st_fpca_miss_test<fdapde::monolithic>(lambda_D,lambda_T,t_steps);
    }
    return 0;
}
