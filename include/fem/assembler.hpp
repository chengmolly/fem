#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <utility>
#include "../mesh/mesh.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/vector.hpp"
#include "shape_function.hpp"
#include "quadrature.hpp"
#include "element_matrix.hpp"

namespace fem
{
    enum class ProblemType
    {
        Poisson,
        LinearElasticity,
        HeatConduction,
        NavierStokes
    };

    struct PDEData
    {
        ProblemType problem_type;
        size_t num_equs;

        std::function<double(const std::vector<double> &)> diffusion_coef;
        std::function<double(const std::vector<double> &)> source_term;
        std::function<std::vector<double>(const std::vector<double> &)> body_force;
        std::function<double(const std::vector<double> &, int, int)> constitutive;

        PDEData() : problem_type(ProblemType::Poisson), num_equs(1) {}
    };

    class Assembler
    {
    public:
        Assembler() : mesh_(nullptr), pde_data_(nullptr), n_dofs_per_node_(1), quad_order_(1) {}

        void init(Mesh &mesh, const PDEData &pde_data, size_t n_dofs_per_node = 1)
        {
            mesh_ = &mesh;
            pde_data_ = &pde_data;
            n_dofs_per_node_ = n_dofs_per_node;

            int dim = mesh.node(0).dim();
            if (dim == 1)
            {
                shape_func_ = std::make_unique<ShapeFunctionLine>(
                    ElementInfo::get_order(mesh.element(0).type()));
            }
            else if (dim == 2)
            {
                if (mesh.element(0).type() == ElementType::TRI3)
                {
                    shape_func_ = std::make_unique<ShapeFunctionTri>(
                        ElementInfo::get_order(mesh.element(0).type()));
                }
                else
                {
                    shape_func_ = std::make_unique<ShapeFunctionQuad>(
                        ElementInfo::get_order(mesh.element(0).type()));
                }
            }
            else
            {
                if (mesh.element(0).type() == ElementType::TET4)
                {
                    shape_func_ = std::make_unique<ShapeFunctionTet>(
                        ElementInfo::get_order(mesh.element(0).type()));
                }
                else
                {
                    shape_func_ = std::make_unique<ShapeFunctionHex>(
                        ElementInfo::get_order(mesh.element(0).type()));
                }
            }

            quad_rule_ = QuadratureFactory::create(mesh.element(0).type(), quad_order_);
        }

        size_t estimaze_nnz() const
        {
            size_t num_elem = static_cast<int>(mesh_->num_elements());
            size_t n_nodes_per_elem = mesh_->element(0).n_nodes();
            size_t n_dofs = n_dofs_per_node_ * n_nodes_per_elem;
            size_t nnz = num_elem * n_dofs * n_dofs;
            return nnz;
        }

        void prepare_matrix_structure(SparseMatrixCRS &A) const
        {
            size_t n_nodes = static_cast<size_t>(mesh_->num_nodes());
            size_t n_dofs = n_nodes * n_dofs_per_node_;

            std::vector<size_t> row_nnz(n_dofs, 0);

            for (size_t e = 0; e < mesh_->num_elements(); ++e)
            {
                const auto &elem = mesh_->element(e);
                const auto &node_ids = elem.node_ids();
                int n_nodes_elem = elem.n_nodes();

                for (size_t i = 0; i < n_nodes_elem; ++i)
                {
                    for (size_t j = 0; j < n_nodes_elem; ++j)
                    {
                        int row = node_ids[i] * n_dofs_per_node_;
                        int col = node_ids[j] * n_dofs_per_node_;

                        for (size_t ii = 0; ii < n_dofs_per_node_; ++ii)
                        {
                            for (size_t jj = 0; jj < n_dofs_per_node_; ++jj)
                            {
                                ++row_nnz[row + ii];
                            }
                        }
                    }
                }
            }

            std::vector<size_t> row_ptr(n_dofs + 1, 0);
            for (size_t i = 0; i < n_dofs; ++i)
            {
                row_ptr[i + 1] = row_ptr[i] + row_nnz[i];
            }

            size_t total_nnz = row_ptr[n_dofs];
            std::vector<size_t> col_idx(total_nnz);
            std::vector<double> values(total_nnz, 0.0);

            size_t idx = 0;
            for (size_t e = 0; e < mesh_->num_elements(); ++e)
            {
                const auto &elem = mesh_->element(e);
                const auto &node_ids = elem.node_ids();
                int n_nodes_elem = elem.n_nodes();

                for (size_t i = 0; i < n_nodes_elem; ++i)
                {
                    for (size_t j = 0; j < n_nodes_elem; ++j)
                    {
                        int row = node_ids[i] * n_dofs_per_node_;
                        int col = node_ids[j] * n_dofs_per_node_;

                        for (size_t ii = 0; ii < n_dofs_per_node_; ++ii)
                        {
                            col_idx[row_ptr[row + ii] + idx] = col + ii;
                        }
                    }
                }
            }

            A.build(n_dofs, n_dofs, std::move(row_ptr), std::move(col_idx), std::move(values));
        }

        void prepare_vector(std::vector<double> &v, double init_val = 0.0) const
        {
            v.assign(mesh_->num_nodes() * n_dofs_per_node_, init_val);
        }

        void assemble(SparseMatrixCRS &A, std::vector<double> &b) const
        {
            size_t n_nodes = static_cast<size_t>(mesh_->num_nodes());
            size_t n_dofs = n_nodes * n_dofs_per_node_;

            A = SparseMatrixCRS(n_dofs, n_dofs);
            // b.assign(n_dofs,0.0);
            prepare_vector(b, 0.0);

            prepare_matrix_structure(A);

            SparseMatrixCOO coo_matrix(n_dofs, n_dofs);
            coo_matrix.reserve(estimaze_nnz());

            for (size_t e = 0; e < mesh_->num_elements(); ++e)
            {
                assemble_element(e, coo_matrix, b);
            }

            A = coo_matrix.to_crs();
        }

        void assemble_element(size_t elem_id, SparseMatrixCOO &A, std::vector<double> &b) const
        {

            const auto &elem = mesh_->element(elem_id);
            const auto &node_ids = elem.node_ids();
            int n_nodes_elem = elem.n_nodes();
            int n_dofs_elem = n_nodes_elem * n_dofs_per_node_;

            std::vector<std::vector<double>> node_coords(n_nodes_elem);
            for (size_t i = 0; i < n_nodes_elem; ++i)
            {
                node_coords[i] = mesh_->node(node_ids[i]).coords();
            }

            auto [Ke, fe] = compute_element_matrix(elem, node_coords);

            for (size_t i = 0; i < n_nodes_elem; ++i)
            {
                size_t global_i = node_ids[i];

                for (size_t di = 0; di < n_dofs_per_node_; ++di)
                {
                    size_t dof_i = global_i * n_dofs_per_node_ + di;
                    b[dof_i] += fe[i * n_dofs_per_node_ + di];
                }

                for (size_t j = 0; j < n_nodes_elem; ++j)
                {
                    size_t global_j = node_ids[j];

                    for (size_t di = 0; di < n_dofs_per_node_; ++di)
                    {
                        for (size_t dj = 0; dj < n_dofs_per_node_; ++dj)
                        {
                            size_t dof_i = global_i * n_dofs_per_node_ + di;
                            size_t dof_j = global_j * n_dofs_per_node_ + dj;
                            size_t local_i = i * n_dofs_per_node_ + di;
                            size_t local_j = j * n_dofs_per_node_ + dj;

                            A.add_uique(dof_i, dof_j, Ke[local_i * n_dofs_elem + local_j]);
                        }
                    }
                }
            }
        }

        std::pair<std::vector<double>, std::vector<double>>
        compute_element_matrix(const Element &elem,
                               const std::vector<std::vector<double>> &node_coords) const
        {
            int n_nodes = elem.n_nodes();
            size_t n_dofs_elem = n_nodes * n_dofs_per_node_;

            std::vector<double> Ke(n_dofs_elem * n_dofs_elem, 0.0);
            std::vector<double> fe(n_dofs_elem, 0.0);

            const auto &quad = *quad_rule_;
            size_t n_quad = quad.num_points();
            const auto &quad_pts = quad.points();
            const auto &quad_wts = quad.weights();

            const auto &shape = *shape_func_;

            for (size_t q = 0; q < n_quad; ++q)
            {
                const auto &xi = quad_pts[q];

                auto N = shape.shape_values(xi);
                std::vector<double> x_phys(node_coords[0].size(), 0.0);
                for (size_t i = 0; i < n_nodes; ++i)
                {
                    for (size_t d = 0; d < x_phys.size(); ++d)
                    {
                        x_phys[d] += N[i] * node_coords[i][d];
                    }
                }

                auto dN = shape.shape_grads(xi);
                std::vector<std::vector<double>> J(node_coords[0].size(),
                                                   std::vector<double>(xi.size(), 0.0));

                for (size_t i = 0; i < n_nodes; ++i)
                {
                    for (size_t d1 = 0; d1 < J.size(); ++d1)
                    {
                        for (size_t d2 = 0; d2 < dN[i].size(); ++d2)
                        {
                            J[d1][d2] += node_coords[i][d1] * dN[i][d2];
                        }
                    }
                }

                double detJ = compute_determinant(J);
                if (std::abs(detJ) < 1e-12)
                    continue;

                auto invJ = compute_inverse(J);

                std::vector<std::vector<double>> dN_phys(n_nodes);
                for (size_t i = 0; i < n_nodes; ++i)
                {
                    dN_phys[i].resize(node_coords[0].size(), 0.0);
                    for (size_t d = 0; d < invJ.size(); ++d)
                    {
                        for (size_t k = 0; k < dN[i].size(); ++k)
                        {
                            dN_phys[i][d] += invJ[d][k] * dN[i][k];
                        }
                    }
                }

                double w = quad_wts[q] * std::abs(detJ);

                if (pde_data_->problem_type == ProblemType::Poisson)
                {
                    double k = 1.0;
                    if (pde_data_->diffusion_coef)
                    {
                        k = pde_data_->diffusion_coef(x_phys);
                    }

                    for (size_t i = 0; i < n_nodes; ++i)
                    {
                        for (size_t j = 0; j < n_nodes; ++j)
                        {
                            double sum = 0.0;
                            for (size_t d = 0; d < dN_phys[i].size(); ++d)
                            {
                                sum += dN_phys[i][d] * dN_phys[j][d];
                            }
                            Ke[i * n_dofs_elem + j] += k * sum * w;
                        }
                    }

                    double f = 0.0;
                    if (pde_data_->source_term)
                    {
                        f = pde_data_->source_term(x_phys);
                    }
                    for (size_t i = 0; i < n_nodes; ++i)
                    {
                        fe[i] += N[i] * f * w;
                    }
                }
            }

            return {Ke, fe};
        }

        double compute_determinant(const std::vector<std::vector<double>> &J) const
        {
            size_t n = static_cast<ssize_t>(J.size());

            if (n == 1)
                return J[0][0];
            if (n == 2)
            {
                return J[0][0] * J[1][1] - J[0][1] * J[1][0];
            }
            if (n == 3)
            {
                return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                       J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                       J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
            }
            return 0.0;
        }

        std::vector<std::vector<double>>
        compute_inverse(const std::vector<std::vector<double>> &J) const
        {
            size_t n = static_cast<size_t>(J.size());
            std::vector<std::vector<double>> invJ(n);

            if (n == 1)
            {
                invJ[0].resize(1);
                invJ[0][0] = 1.0 / J[0][0];
            }
            else if (n == 2)
            {
                double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
                invJ[0] = {J[1][1] / det, -J[0][1] / det};
                invJ[1] = {-J[1][0] / det, J[0][0] / det};
            }
            else if (n == 3)
            {
                double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                             J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                             J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
                invJ[0] = {(J[1][1] * J[2][2] - J[1][2] * J[2][1]) / det,
                           (J[0][2] * J[2][1] - J[0][1] * J[2][2]) / det,
                           (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / det};
                invJ[1] = {(J[1][2] * J[2][0] - J[1][0] * J[2][2]) / det,
                           (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / det,
                           (J[0][2] * J[1][0] - J[0][0] * J[1][2]) / det};
                invJ[2] = {(J[1][0] * J[2][1] - J[1][1] * J[2][0]) / det,
                           (J[0][1] * J[2][0] - J[0][0] * J[2][1]) / det,
                           (J[0][0] * J[1][1] - J[0][1] * J[1][0]) / det};
            }
            return invJ;
        }

    private:
        Mesh *mesh_;
        const PDEData *pde_data_;
        size_t n_dofs_per_node_;
        size_t quad_order_;

        std::unique_ptr<ShapeFunction> shape_func_;
        std::unique_ptr<QuadratureRule> quad_rule_;
    };
} // namespace fem