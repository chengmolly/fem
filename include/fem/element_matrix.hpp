#pragma once

#include <vector>
#include <functional>
#include "../core/matrix.hpp"
#include "../mesh/mesh.hpp"
#include "shape_function.hpp"
#include "quadrature.hpp"

namespace fem
{

    struct ElementMatrixCalculator
    {
        static DenseMatrix<double>
        compute_stiffness_matrix(const Element &elem,
                                 const std::vector<std::vector<double>> &node_coords,
                                 const std::vector<std::vector<double>> &dN_phys,
                                 double detJ,
                                 const std::vector<double> &weights)
        {
            size_t num_nodes = elem.n_nodes();
            DenseMatrix<double> Ke(num_nodes, num_nodes);
            Ke.zero();

            for (size_t q = 0; q < weights.size(); ++q)
            {
                double w = weights[q] * std::abs(detJ);

                for (size_t i = 0; i < num_nodes; ++i)
                {
                    for (size_t j; j < num_nodes; ++j)
                    {
                        double sum = 0.0;
                        for (size_t d = 0; d < dN_phys[i].size(); ++d)
                        {
                            sum += dN_phys[i][d] * dN_phys[j][d];
                        }
                        Ke(i, j) += sum * w;
                    }
                }
            }
            return Ke;
        }

        static DenseMatrix<double>
        compute_mass_matrix(const Element &elem,
                            const std::vector<std::vector<double>> &node_coords,
                            const std::vector<double> &N,
                            double detJ,
                            const std::vector<double> &weights,
                            double rho = 1.0)
        {
            size_t num_nodes = elem.n_nodes();
            DenseMatrix<double> Me(num_nodes, num_nodes);
            Me.zero();

            for (size_t q = 0; q < weights.size(); ++q)
            {
                double w = weights[q] * std::abs(detJ);

                for (size_t i = 0; i < num_nodes; ++i)
                {
                    for (size_t j = 0; j < num_nodes; ++j)
                    {
                        Me(i, j) += N[i] * N[j] * w * rho;
                    }
                }
            }
            return Me;
        }

        static std::vector<double>
        compute_load_vector(const Element &elem,
                            const std::vector<std::vector<double>> &node_coords,
                            const std::vector<double> &N,
                            double detJ,
                            const std::vector<double> &weights,
                            std::function<double(const std::vector<double> &)> f = nullptr)
        {
            size_t num_nodes = elem.n_nodes();
            std::vector<double> fe(num_nodes, 0.0);
            for (size_t q = 0; q < weights.size(); ++q)
            {
                double w = weights[q] * std::abs(detJ);

                std::vector<double> x_phys(node_coords[0].size(), 0.0);
                for (size_t i = 0; i < num_nodes; ++i)
                {
                    for (size_t d = 0; d < x_phys.size(); ++d)
                    {
                        x_phys[d] += N[i] * node_coords[i][d];
                    }
                }

                double f_val = f ? f(x_phys) : 1.0;

                for (size_t i = 0; i < num_nodes; ++i)
                {
                    fe[i] += N[i] * f_val * w;
                }
            }
            return fe;
        }

        static std::vector<double>
        compute_boundary_flux(const Element &elem,
                              const std::vector<std::vector<double>> &node_coords,
                              const std::vector<double> &N,
                              const std::vector<std::vector<double>> &dN_phys,
                              const std::vector<std::vector<double>> &quad_pts,
                              const std::vector<double> &weights,
                              const std::vector<double> &normal,
                              std::function<double(const std::vector<double> &)> flux = nullptr)
        {
            size_t num_nodes = elem.n_nodes();
            std::vector<double> fe(num_nodes, 0.0);

            for (size_t q = 0; q < weights.size(); ++q)
            {
                double w = weights[q];

                std::vector<double> x_phys(node_coords[0].size(), 0.0);
                for (size_t i = 0; i < num_nodes; ++i)
                {
                    for (size_t d = 0; d < x_phys.size(); ++d)
                    {
                        x_phys[d] += N[i] * node_coords[i][d];
                    }
                }

                double flux_val = flux ? flux(x_phys) : 1.0;

                for (size_t i = 0; i < num_nodes; ++i)
                {
                    fe[i] += N[i] * flux_val * w;
                }
            }
            return fe;
        }
    };

    struct ElastictyElementMatrix
    {
    };
} // namespace fem