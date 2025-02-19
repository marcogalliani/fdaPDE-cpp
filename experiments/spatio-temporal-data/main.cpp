//
// Created by Marco Galliani on 19/12/24.
//

#include <Eigen/Eigenvalues>

#include "nlohmann/json.hpp"
using nlohmann::json;

#include "experiments/cpp-utils/st-data-generation.cpp"

int main(){
    //parameters
    std::ifstream input("params.json");
    json data = json::parse(input);

    //EVD solver
    std::string evd_type = data["RunParams"].value("EVDType","exact");

    //Diffusion operator params
    double alpha1 = data["DiffusionOp"].value("alpha1",0.0);
    double alpha2 = data["DiffusionOp"].value("alpha2",0.0);
    double gamma1 = data["DiffusionOp"].value("gamma1",0.0);
    double gamma2 = data["DiffusionOp"].value("gamma2",0.0);

    //number of eigenvectors
    int n_eigenvects = data["RunParams"].value("n_eigenvectors",4);

    if (evd_type == "randomized"){
        generate_st_components<REVD<SpMatrix<double>>>(n_eigenvects,alpha1,alpha2,gamma1,gamma2);
    }else if(evd_type == "exact"){
        generate_st_components<Eigen::SelfAdjointEigenSolver<DMatrix<double>>>(n_eigenvects,alpha1,alpha2,gamma1,gamma2);
    }
}