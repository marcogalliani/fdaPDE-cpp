#include <gtest/gtest.h> // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>

// regression test suite
/*
#include "src/srpde_test.cpp"
#include "src/strpde_test.cpp"
#include "src/gsrpde_test.cpp"
#include "src/qsrpde_test.cpp"
#include "src/gcv_srpde_test.cpp"
#include "src/gcv_qsrpde_test.cpp"
#include "src/gcv_srpde_newton_test.cpp"
 */
// #include "src/kcv_srpde_test.cpp"
// functional test suite
//#include "src/fpca_test.cpp"
#include "src/fpca_extensive_testing.cpp"
/*
#include "src/fpls_test.cpp"
#include "src/centering_test.cpp"
 */

using fdapde::core::RSI;
using fdapde::core::RBKI;

int main(int argc, char **argv){

    // SVD solvers
    RSI<DMatrix<double>,fdapde::core::extended> svd_rsi(3,3); //1st: rank, 2nd: oversampling
    RBKI<DMatrix<double>,fdapde::core::extended> svd_rbki(3,10); //1st: rank, 2nd: block size
    Eigen::JacobiSVD<DMatrix<double>> exact_svd(0,0,Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Monolithic RSVD
    //fpca_generated_test<Eigen::JacobiSVD<DMatrix<double>>,fdapde::monolithic>(exact_svd);

    // Sequential RSVD
    //fpca_generated_test<RSI<DMatrix<double>,fdapde::core::extended>,fdapde::sequential>(svd_rsi);
    fpca_generated_test<Eigen::JacobiSVD<DMatrix<double>>,fdapde::sequential>(exact_svd);
    //fpca_library_test<RSI<DMatrix<double>,fdapde::core::extended> ,fdapde::sequential>(svd_rsi);
    /*
    // start testing
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    */

     return 0;
}
