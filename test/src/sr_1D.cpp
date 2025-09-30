// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "fdaPDE/io.h"
#include "fdaPDE/src/solvers/sp_smoothing.h"
using namespace fdapde;
using fdapde::test::almost_equal;

// test 1
//    mesh:         unit_interval
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(sr, test_01) {
    // geometry
    //Eigen::MatrixXd knots = read_table<double>("../data/sr/1D_test1/knots.csv").as_matrix();
    Triangulation<1,1> T = Triangulation<1,1>::UnitInterval(101);
    // data
    GeoFrame data(T);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", T.nodes());
    l1.load_csv<double>("../data/sr/1D_test1/y.csv");
    // physics
    BsSpace Bh(T, 3);   // cubic B-splines
    TrialFunction f_t(Bh);
    TestFunction  v_t(Bh);
    auto a_t = integral(T)(dxx(f_t) * dxx(v_t));  // curvature penalty
    ZeroField<1> u_t;
    auto F_t = integral(T)(u_t * v_t);
    // modeling
    SRPDE m("x ~ f", data, sp_smoothing(a_t, F_t));
    m.fit(1e-4/101);

    EXPECT_TRUE(almost_equal<double>(m.f(), "../data/sr/1D_test1/sol.mtx"));
}

// test 1
//    mesh:         unit_interval
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(sr, test_02) {
    // geometry
    //Eigen::MatrixXd knots = read_table<double>("../data/sr/1D_test1/knots.csv").as_matrix();
    Triangulation<1,1> T = Triangulation<1,1>::UnitInterval(101);
    // data
    GeoFrame data(T);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", T.nodes());
    l1.load_csv<double>("../data/sr/1D_test1/y.csv");
    // physics
    BsSpace Bh(T, 3);   // cubic B-splines
    TrialFunction f_t(Bh);
    TestFunction  v_t(Bh);
    auto a_t = integral(T)(dxx(f_t) * dxx(v_t));  // curvature penalty
    ZeroField<1> u_t;
    auto F_t = integral(T)(u_t * v_t);
    // modeling
    SRPDE m("x ~ f", data, sp_smoothing(a_t, F_t));

    // calibration
    std::vector<double> lambda_grid;
    for (double x = -6.0; x <= 2.0; x += 1) lambda_grid.push_back(std::pow(10, x)/101);
    GridOptimizer<1> optimizer;
    optimizer.optimize(m.gcv(1e2, 476813), lambda_grid);
    //for (int i=0; i < lambda_grid.size(); i++) { std::cout << optimizer.values()[i] << "\n";}
    //EXPECT_TRUE(almost_equal<double>(optimizer.values(), "../data/sr/1D_test1/gcv_scores.mtx"));
    // final fit
    m.fit(optimizer.optimum());
    EXPECT_TRUE(almost_equal<double>(m.f(), "../data/sr/1D_test1/sol_kcv.mtx"));
}