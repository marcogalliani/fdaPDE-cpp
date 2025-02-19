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
    double alpha = data["DiffusionOp"].value("alpha",0.0);
    double gamma = data["DiffusionOp"].value("gamma",0.0);

    //number of eigenvectors
    int n_eigenvects = data["RunParams"].value("n_eigenvectors",4);

    double tol = data["RunParams"].value("tol",0.001);
    int max_iter = data["RunParams"].value("max_iter",1e3);

    if(evd_type == "randomized"){
        generate_spatial_components<REVD<SpMatrix<double>>>(n_eigenvects,alpha,gamma,tol,max_iter);

    }else if(evd_type == "exact"){
        generate_spatial_components<Eigen::SelfAdjointEigenSolver<DMatrix<double>>>(n_eigenvects,alpha,gamma);
    }
}