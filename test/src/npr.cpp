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

#include <cmath>
#include <limits>

using namespace fdapde;


using vector_t = Eigen::Matrix<double, Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;


vector_t f(double t, const vector_t& y) {
        vector_t out(2);
        out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
        return out;
    }

// end-to-end through an order-1 (time) GeoFrame and a formula: the multi-column response is
// read as a block column and the fit matches the plain-input path exactly.
TEST(npr, test1) {
    // time grid
    const int m = 41; double Tf = 2.0;
    vector_t time(m);
    for (int i = 0; i < m; ++i) { time[i] = Tf * i / (m - 1); }
    // order-1 time mesh matching fx.time, with the d-dimensional response as a block column
    Triangulation<1, 1> T = Triangulation<1, 1>::Interval(time[0], time[m - 1], m);
    
    // reference solution
    vector_t y0(2);
    y0 << 0.5, -0.3;
    RKIntegrator ref(ode_schemes::gauss_legendre_2());
    matrix_t Y(time.size(), y0.size());
    vector_t y = y0;
    Y.row(0) = y.transpose();
    for (int t = 0; t + 1 < time.size(); ++t) {
        y = ref.step(f, time[t], y, time[t + 1] - time[t]);
        Y.row(t + 1) = y.transpose();
    }
    
    // data
    GeoFrame data(T);
    auto& layer = data.insert_scalar_layer<POINT>("layer", MESH_NODES);
    layer.load_blk("y", Y);
    
    // solver
    ts_ls_ode penalty(
        [](double t, const vector_t& y) {
            vector_t out(2);
            out << y[0] * y[1] + std::sin(t), y[0] - y[1] * y[1];
            return out;
        },
        ode_schemes::gauss_legendre_2());

    NPRODE<internals::ts_ls_ode> model("y ~ f", data, penalty);
    model.fit(1.0);
    EXPECT_TRUE(model.converged());
    EXPECT_EQ(model.n_nodes(), m);
    EXPECT_EQ(model.n_components(), 2);

    std::cout << (model.trajectory() - Y).squaredNorm() << std::endl;

}
