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
    double fixed_lambda = data["RunParams"].value("fixed_lambda",1e-1);

    std::vector<double> lambda_PCs = data["SubspaceIterations"]["lambdas"].get<std::vector<double>>();
    int max_iter_subspace = data["SubspaceIterations"].value("max_iter",30);

    if(rsvd_version == "sequential") fpca_test<fdapde::sequential>(fixed_lambda);
    else if(rsvd_version == "monolithic") fpca_test<fdapde::monolithic>(fixed_lambda);
    else if(rsvd_version == "subspace") fpca_subspace_fixed(fixed_lambda,max_iter_subspace);

    return 0;
}
