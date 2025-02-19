//
// Created by Marco Galliani on 19/12/24.
//

#include <fstream>
#include <iostream>
#include <cstddef>
#include <chrono>

#include "fdaPDE/core.h"
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::diffusion;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

using fdapde::core::REVD;
using fdapde::core::is_rand_evd;

#include "test/src/utils/constants.h"
#include "test/src/utils/mesh_loader.h"
#include "test/src/utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

#include <Eigen/SVD>

template <typename EVDType>
void generate_spatial_components(int n_eigvects, double alpha, double gamma, double tol=1e-3, int max_iter=1e3){
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    //regularizing PDE
    Eigen::Matrix<double,2,2> R;
    R << std::cos(alpha), -std::sin(alpha),
            std::sin(alpha), std::cos(alpha);
    DVector<double> Sigma(2);
    Sigma << 1/std::sqrt(gamma), std::sqrt(gamma);

    Eigen::Matrix<double,2,2> K = R * Sigma.asDiagonal() * R.transpose();

    auto L = -diffusion<FEM,Eigen::Matrix<double,2,2>>(K);

    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);

    pde.init();

    Eigen::saveMarket(pde.mass(),"results/R0.mtx");
    Eigen::saveMarket(pde.stiff(),"results/R1.mtx");

    EVDType evd;
    const auto start{std::chrono::steady_clock::now()};
    if constexpr (is_rand_evd<EVDType>{}){
        evd.setTol(tol);
        evd.compute(pde.stiff(),n_eigvects,max_iter);
        Eigen::saveMarket(evd.matrixU(),"results/eigenvectorsR1.mtx");
        Eigen::saveMarket(evd.eigenValues(),"results/eigenvaluesR1.mtx");
    }else{
        evd.compute(pde.stiff());
        Eigen::saveMarket(evd.eigenvectors().rowwise().reverse(),"results/eigenvectorsR1.mtx");
        Eigen::saveMarket(evd.eigenvalues().reverse(),"results/eigenvaluesR1.mtx");
    }
    const auto end{std::chrono::steady_clock::now()};

    std::ofstream test_report("results/test_report.csv");
    test_report << (std::chrono::duration<double>{end - start}).count() << std::endl;
    test_report.close();
    return;
};

template <typename EVDType>
void generate_st_components(int n_eigvects, double alpha1, double alpha2, double gamma1, double gamma2){
    MeshLoader<Triangulation<3, 3>> domain("unit_cube");
    //regularizing PDE
    Eigen::Matrix<double,3,3> R1,R2,R;
    R1 << std::cos(alpha1), -std::sin(alpha1), 0.0,
            std::sin(alpha1), std::cos(alpha1), 0.0,
            0.0, 0.0, 1.0;
    R2 << std::cos(alpha2), 0.0, std::sin(alpha2),
            0.0, 1.0, 0.0,
            -std::sin(alpha2), 0.0, std::cos(alpha2);
    R = R1*R2;

    DVector<double> Sigma(3);
    Sigma << std::cbrt(std::pow(gamma1,2)/gamma2),
            std::cbrt(std::pow(gamma2,2)/gamma1),
            std::cbrt(1/gamma1*gamma2);

    Eigen::Matrix<double,3,3> K = R * Sigma.asDiagonal() * R.transpose();

    auto L = -diffusion<FEM,Eigen::Matrix<double,3,3>>(K);

    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 4, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);

    pde.init();

    Eigen::saveMarket(pde.mass(),"results/R0.mtx");

    EVDType evd;
    if constexpr (is_rand_evd<EVDType>{}){
        evd.compute(pde.stiff(),n_eigvects);
        Eigen::saveMarket(evd.matrixU(),"results/eigenvectorR1.mtx");
    }else{
        evd.compute(pde.stiff());
        Eigen::saveMarket(evd.eigenvectors(),"results/eigenvectorR1.mtx");
    }
    return;
};



