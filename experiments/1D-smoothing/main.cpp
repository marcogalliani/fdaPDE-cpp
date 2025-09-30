//
// Created by Marco Galliani on 21/05/25.
// Enriched with noise and error evaluation
//
#include <fdaPDE/models.h>
#include "fdaPDE/src/solvers/sp_smoothing.h"
using namespace fdapde;

#include <random>
#include <iostream>
#include <cmath>

// Utility: align columns by sign
Eigen::MatrixXd align_columns_by_sign(const Eigen::MatrixXd& reference, Eigen::MatrixXd target) {
    for (int col = 0; col < reference.cols(); ++col) {
        double dot_product = reference.col(col).dot(target.col(col));
        if (dot_product < 0) target.col(col) *= -1;
    }
    return target;
}

// Three orthogonal functions on [0,1]
inline double f1(double t) { return 1.0; }
inline double f2(double t) { return std::sqrt(3.0) * (2.0 * t - 1.0); }
inline double f3(double t) { return std::sqrt(5.0) * (6.0 * t * t - 6.0 * t + 1.0); }

int main() {
    // simulation parameters
    int n_units = 50;   // number of curves
    int n_points = 100; // discretization of [0,1]

    // mesh: uniform grid on [0,1]
    Eigen::MatrixXd nodes(n_points, 1);
    for (int i = 0; i < n_points; ++i) nodes(i, 0) = static_cast<double>(i) / (n_points - 1);
    Triangulation<1, 1> T(nodes);

    // geoframe with one point layer
    GeoFrame data(T);
    auto& l = data.insert_scalar_layer<POINT>("layer", nodes);

    // generate synthetic data
    Eigen::MatrixXd Y(n_points, n_units);
    std::mt19937 gen(42);
    std::normal_distribution<double> dist1(0.0, 1.0);
    std::normal_distribution<double> dist2(0.0, 0.5);
    std::normal_distribution<double> dist3(0.0, 0.25);
    std::normal_distribution<double> noise(0.0, 0.05); // Gaussian noise

    for (int j = 0; j < n_units; ++j) {
        double a1 = dist1(gen), a2 = dist2(gen), a3 = dist3(gen);
        for (int i = 0; i < n_points; ++i) {
            double t = nodes(i, 0);
            Y(i, j) = a1 * f1(t) + a2 * f2(t) + a3 * f3(t) + noise(gen); // add noise
        }
        l.load_vec("x" + std::to_string(j + 1), Y.col(j));
    }
    l.data().merge<double>("X");

    // Physics: cubic B-splines with roughness penalty
    BsSpace Bh(T, 3);   // cubic B-splines
    TrialFunction f_t(Bh);
    TestFunction  v_t(Bh);
    auto a_t = integral(T)(dxx(f_t) * dxx(v_t));  // curvature penalty
    ZeroField<1> u_t;
    auto F_t = integral(T)(u_t * v_t);

    // ---------------------------------------------------------
    // FPCA with smoothing
    // ---------------------------------------------------------
    fPCA m("X", data, sp_smoothing(a_t, F_t));
    std::vector<double> lambda_grid = {1e-6, 1e-5, 1e-4, 1e-3};

    m.fit(
        /* n_comp = */ 3,
        lambda_grid,
        /* options = */ OptimizeMSRE | ComputeRandSVD,
        fpca_subspace_solver()
    );

    // ---------------------------------------------------------
    // Output
    // ---------------------------------------------------------
    std::cout << "Selected lambda: " << m.lambda().transpose() << "\n";
    std::cout << "Loadings (Fn):\n" << m.Fn().leftCols(3) << "\n";
    std::cout << "Scores (S):\n"   << m.S().topRows(5) << "\n";

    // ---------------------------------------------------------
    // Compute reconstruction error
    // ---------------------------------------------------------
    Eigen::MatrixXd Yhat = m.Fn() * m.S().transpose(); // reconstructed curves
    Yhat = align_columns_by_sign(Y, Yhat);

    double mse = (Y - Yhat).squaredNorm() / (Y.rows() * Y.cols());
    double rel_error = (Y - Yhat).norm() / Y.norm();

    std::cout << "MSE of reconstruction: " << mse << "\n";
    std::cout << "Relative error of reconstruction: " << rel_error << "\n";

    return 0;
}
