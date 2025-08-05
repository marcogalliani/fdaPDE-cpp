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

using namespace fdapde;
using fdapde::test::almost_equal;
using fdapde::write_csv;

using vector_t = Eigen::Matrix<double, Dynamic, 1>;
using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
using sparse_matrix_t = Eigen::SparseMatrix<double>;
using binary_t = BinaryMatrix<Dynamic, Dynamic>;

// test 1
//    mesh:         unit_square_60
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
TEST(fpca, test_01) {
    // geometry
    std::string mesh_path = "../data/mesh/unit_square_10/";
    Triangulation<2, 2> D(mesh_path + "/points.csv", mesh_path + "/elements.csv", mesh_path + "/boundary.csv",
        /* header = */ true, /* index_col = */ true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    std::string data_path = "../data/fpca/01/";
    matrix_t X = read_csv<double>(data_path + "y.csv").as_matrix();
    l1.load_blk("X", X.transpose());
    // physics (isotropic laplacian)
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    fPCA fpca("X", data, fe_ls_elliptic(a, F));
    // fit
    std::vector<double> lambda_grid = {1e-5}; // if the grid contains more than a value the calibration is automatically activated
    fpca.fit(
        /* n_comp = */ 3,
        lambda_grid,
        /* options = */ ComputeRandSVD, // to use calibration (only kcv for now) replace with: ComputeRandSVD | OptimizeMSRE
        fpca_subspace_solver()
    );

    // results
    matrix_t X_reconstructed = fpca.S()*fpca.Fn().transpose();
    X_reconstructed = X_reconstructed.rowwise() + fpca.center_locs().transpose(); // add the estimated functional mean
    matrix_t X_true = read_csv<double>(data_path + "true_reconstruction.csv",true, false).as_matrix();
    EXPECT_TRUE(almost_equal<double>(X_reconstructed, X_true));
}