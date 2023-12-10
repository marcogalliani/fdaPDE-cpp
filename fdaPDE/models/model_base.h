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

#ifndef __MODEL_BASE_H__
#define __MODEL_BASE_H__

#include <fdaPDE/mesh.h>
#include <fdaPDE/utils.h>
#include <fdaPDE/pde.h>
using fdapde::core::BlockFrame;
using fdapde::core::pde_ptr;

#include "model_macros.h"
#include "model_traits.h"
#include "model_runtime.h"

namespace fdapde {
namespace models {

// abstract base interface for any fdaPDE statistical model.
template <typename Model> class ModelBase {
   public:
    // constructors
    ModelBase() = default;
    ModelBase(const pde_ptr& pde) : pde_(pde) { pde_.init(); };
    // full model stack initialization
    void init() {
        if (model().runtime().query(runtime_status::require_pde_init)) { pde_.init(); }   // init penalty
        if (model().runtime().query(runtime_status::require_penalty_init)) { model().init_regularization(); }        
        if (model().runtime().query(runtime_status::require_functional_basis_evaluation)) {
            model().init_sampling(true);   // init \Psi matrix, always force recomputation
            model().init_nan();            // analyze and set missingness pattern
        }
        
        model().init_data();    // specific data-dependent initialization requested by the model
        model().init_model();   // model initialization
    }

    // setters
    void set_data(const BlockFrame<double, int>& df, bool reindex = false) {
        df_ = df;
        // insert an index row (if not yet present or requested)
        if (!df_.has_block(INDEXES_BLK) || reindex) {
            std::size_t n = df_.rows();
            DMatrix<int> idx(n, 1);
            for (std::size_t i = 0; i < n; ++i) idx(i, 0) = i;
            df_.insert(INDEXES_BLK, idx);
        }
    }
    void set_pde(const pde_ptr& pde) {
        pde_ = pde;
        model().runtime().set(runtime_status::require_pde_init);
    }
    void set_lambda(const DVector<double>& lambda) {   // dynamic sized version of set_lambda provided by upper layers
	model().set_lambda_D(lambda[0]);
	if constexpr(is_space_time<Model>::value) model().set_lambda_T(lambda[1]);
    }

    // getters
    const BlockFrame<double, int>& data() const { return df_; }
    BlockFrame<double, int>& data() { return df_; }   // direct write-access to model's internal data storage
    const DMatrix<int>& idx() const { return df_.get<int>(INDEXES_BLK); }   // data indices
    const pde_ptr& pde() const { return pde_; }   // regularizing term Lf - u (defined on some domain D)
    std::size_t n_locs() const { return model().n_spatial_locs() * model().n_temporal_locs(); }
    DVector<double> lambda(int) const { return model().lambda(); }   // int supposed to be fdapde::Dynamic
    // access to model runtime status
    model_runtime_handler& runtime() { return runtime_; }
    const model_runtime_handler& runtime() const { return runtime_; }

    virtual ~ModelBase() = default;
   protected:
    pde_ptr pde_ {};                     // regularizing term Lf - u
    BlockFrame<double, int> df_ {};      // blockframe for data storage
    model_runtime_handler runtime_ {};   // model's runtime status

    // getter to underlying model object
    inline Model& model() { return static_cast<Model&>(*this); }
    inline const Model& model() const { return static_cast<const Model&>(*this); }
};

// set boundary conditions on problem's linear system
// BUG: not working - fix needed due to SparseBlockMatrix interface
// template <typename Model> void ModelBase<Model>::set_dirichlet_bc(SpMatrix<double>& A, DMatrix<double>& b) {
//     std::size_t n = A.rows() / 2;

//     for (std::size_t i = 0; i < n; ++i) {
//         if (pde_->domain().is_on_boundary(i)) {
//             A.row(i) *= 0;          // zero all entries of this row
//             A.coeffRef(i, i) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

//             A.row(i + n) *= 0;
//             A.coeffRef(i + n, i + n) = 1;

//             // boundaryDatum is a pair (nodeID, boundary value)
//             double boundaryDatum = pde_->boundaryData().empty() ? 0 : pde_->boundaryData().at(i)[0];
//             b.coeffRef(i, 0) = boundaryDatum;   // impose boundary value
//             b.coeffRef(i + n, 0) = 0;
//         }
//     }
//     return;
// }

}   // namespace models
}   // namespace fdapde

#endif   // __MODEL_BASE_H__
