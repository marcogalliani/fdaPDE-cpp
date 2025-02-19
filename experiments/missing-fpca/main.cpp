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
    std::string svd_policy = data["RunParams"].value("SVDType","sequential");

    int start_rec = data["RunParams"].value("start_recursion",1);
    int n_pc = data["RunParams"].value("n_pc",3);
    double fixed_lambda = data["RunParams"].value("lambda",1e-5);

    if(rsvd_policy == "sequential"){
        if(svd_policy == "exact") fpca_miss_test<fdapde::sequential,Eigen::JacobiSVD<DMatrix<double>>>(fixed_lambda,n_pc,start_rec);
        else if(svd_policy == "randomized") fpca_miss_test<fdapde::sequential>(fixed_lambda,n_pc,start_rec);
    }
    if(rsvd_policy == "monolithic"){
        if(svd_policy == "exact") fpca_miss_test<fdapde::monolithic,Eigen::JacobiSVD<DMatrix<double>>>(fixed_lambda,n_pc,start_rec);
        else if(svd_policy == "randomized") fpca_miss_test<fdapde::monolithic>(fixed_lambda,n_pc,start_rec);
    }

    return 0;
}
