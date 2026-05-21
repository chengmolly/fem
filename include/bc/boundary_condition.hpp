#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <map>
#include "../mesh/mesh.hpp"
#include "../core/vector.hpp"
#include "../core/sparse_matrix.hpp"

namespace fem
{
    enum class BCType
    {
        Dirichlet,
        Neumann,
        Robin
    };

    class BoundaryCondition
    {
    public:
        virtual ~BoundaryCondition() = default;

        virtual BCType type() const = 0;
        virtual void apply(SparseMatrixCRS &A, std::vector<double> &b, std::vector<double> &x) const = 0;
        virtual void apply(std::vector<double> &x) const = 0;
    };

    class DirichletBC : public BoundaryCondition
    {
    public:
        DirichletBC() = default;

        DirichletBC(int node_id, double value)
        {
            values_[node_id] = value;
        }

        DirichletBC(const std::vector<int> &node_ids, const std::vector<double> &values)
        {
            for (size_t i = 0; i < node_ids.size(); ++i)
            {
                values_[node_ids[i]] = values[i];
            }
        }

        DirichletBC(std::vector<int> &&node_ids, std::vector<double> &&values)
        {
            for (size_t i = 0; i < node_ids.size(); ++i)
            {
                values_[node_ids[i]] = values[i];
            }
        }

        BCType type() const override { return BCType::Dirichlet; }

        void add_node(int node_id, double value)
        {
            values_[node_id] = value;
        }

        void remove_node(int node_id)
        {
            values_.erase(node_id);
        }

        bool has_node(int node_id) const
        {
            return values_.find(node_id) != values_.end();
        }

        double get_value(int node_id) const
        {
            auto it = values_.find(node_id);
            if (it == values_.end())
            {
                throw std::runtime_error("node_id not found");
            }
            else
            {
                return it->second;
            }
        }

        const std::map<int, double> &values() const { return values_; }

        void apply(SparseMatrixCRS &A, std::vector<double> &b,
                   std::vector<double> &x) const override
        {
            for (const auto &[node_id, val] : values_)
            {
                if (node_id < static_cast<int>(x.size()))
                {
                    x[node_id] = val;
                }

                const auto &row_ptr = A.row_ptr();
                const auto &col_idx = A.col_idx();
                const auto &values = A.values();

                for (int j = row_ptr[node_id]; j < row_ptr[node_id + 1]; ++j)
                {
                    int col = col_idx[j];

                    if (col == node_id)
                    {
                        // diagonal element set to 1.0
                        const_cast<double &>(values[j]) = 1.0;
                    }
                    else
                    {
                        // non-diagonal element set to 0.0,move to right term
                        const_cast<double &>(b[node_id]) -= values[j] * x[col];
                        const_cast<double &>(values[j]) = 0.0;
                    }
                }

                // set right term
                const_cast<double &>(b[node_id]) = val;
            }
        }

        void apply(std::vector<double> &x) const override
        {
            for (const auto &[node_id, val] : values_)
            {
                if (node_id < static_cast<int>(x.size()))
                {
                    x[node_id] = val;
                }
            }
        }

        void apply_penalty(SparseMatrixCRS &A, std::vector<double> &b, double penalty = 1e15) const
        {
            for (const auto &[node_id, val] : values_)
            {
                const auto &row_ptr = A.row_ptr();
                const auto &col_idx = A.col_idx();
                auto &values = const_cast<std::vector<double> &>(A.values());

                for (int j = row_ptr[node_id]; j < row_ptr[node_id + 1]; ++j)
                {
                    if (col_idx[j] == node_id)
                    {
                        double old_diag = values[j];
                        values[j] = penalty;
                        b[node_id] = penalty * val;

                        for (int i = 0; i < node_id; ++i)
                        {
                            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
                            {
                                if (col_idx[k] == node_id)
                                {
                                    b[i] -= values[k] * (val - b[i]) * (old_diag / penalty);
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        std::vector<int> get_node_ids() const
        {
            std::vector<int> ids;
            for (const auto &[id, _] : values_)
            {
                ids.push_back(id);
            }
            return ids;
        }

    private:
        std::map<int, double> values_;
    };

    class NeumannBC : public BoundaryCondition
    {
    public:
        NeumannBC() = default;

        NeumannBC(int node_id, double flux)
        {
            fluxes_[node_id] = flux;
        }

        NeumannBC(const std::vector<int> &node_ids, const std::vector<double> &fluxes)
        {
            for (size_t i = 0; i < node_ids.size(); ++i)
            {
                fluxes_[node_ids[i]] = fluxes[i];
            }
        }

        BCType type() const override { return BCType::Neumann; }

        void add_flux(int node_id, double flux)
        {
            fluxes_[node_id] += flux;
        }

        void apply(SparseMatrixCRS &, std::vector<double> &b, std::vector<double> &) const override
        {
            for (const auto &[node_id, flux] : fluxes_)
            {
                if (node_id < static_cast<int>(b.size()))
                {
                    b[node_id] += flux;
                }
            }
        }
        void apply(std::vector<double> &) const override {}

    private:
        std::map<int, double> fluxes_;
    };

    class RobinBC : public BoundaryCondition
    {
    public:
        RobinBC() = default;

        RobinBC(int node_id, double alpha, double beta)
        {
            params_[node_id] = {alpha, beta};
        }

        BCType type() const override { return BCType::Robin; }

        void apply(SparseMatrixCRS &A, std::vector<double> &b, std::vector<double> &) const override
        {
            for (const auto &[node_id, params] : params_)
            {
                double alpha = params.first;
                double beta = params.second;

                // modify A and b
            }
        }

        void apply(std::vector<double> &) const override {}

    private:
        std::map<int, std::pair<double, double>> params_;
    };

    class BCManager
    {
    public:
        BCManager() = default;

        void add_dirichlet(const std::vector<int> &nodes, const std::vector<double> &values)
        {
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                dirichlet_bcs_.add_node(nodes[i], values[i]);
            }
        }

        void add_neumann(const std::vector<int> &nodes, const std::vector<double> &fluxes)
        {
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                neumann_bcs_list_.push_back(std::make_shared<NeumannBC>(nodes[i], fluxes[i]));
            }
        }

        void add_robin(int node_id, double alpha, double beta)
        {
            robin_bcs_list_.push_back(std::make_shared<RobinBC>(node_id, alpha, beta));
        }

        void apply(SparseMatrixCRS &A, std::vector<double> &b, std::vector<double> &x) const
        {
            for (const auto &bc : all_bcs_)
            {
                bc->apply(A, b, x);
            }
        }

        void apply_dirichlet(SparseMatrixCRS &A, std::vector<double> &b, std::vector<double> &x) const
        {
            dirichlet_bcs_.apply(A, b, x);
        }

        void apply_neumann(std::vector<double> &b) const
        {
            neumann_bcs_.apply(b);
            for (const auto &bc : neumann_bcs_list_)
            {
                bc->apply(b);
            }
        }

        void apply_to_solution(std::vector<double> &x) const
        {
            dirichlet_bcs_.apply(x);
        }

        void apply_penalty(SparseMatrixCRS &A, std::vector<double> &b, double penalty = 1e15) const
        {
            dirichlet_bcs_.apply_penalty(A, b, penalty);
        }

        bool is_dirichlet(int node_id) const
        {
            return dirichlet_bcs_.has_node(node_id);
        }

        double get_dirichlet_values(int node_id) const
        {
            return dirichlet_bcs_.get_value(node_id);
        }

        template <typename Func>
        void set_dirichlet_from_function(const std::vector<int> &nodes, Func &&func)
        {
            std::vector<double> values(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                values[i] = func(nodes[i]);
            }
            add_dirichlet(nodes, values);
        }

        // mark boundary points
        void mark_boundary_nodes(Mesh &mesh) const;

    private:
        DirichletBC dirichlet_bcs_;
        NeumannBC neumann_bcs_;
        std::vector<std::shared_ptr<NeumannBC>> neumann_bcs_list_;
        std::vector<std::shared_ptr<RobinBC>> robin_bcs_list_;
        std::vector<std::shared_ptr<BoundaryCondition>> all_bcs_;
    };
} // namespace fem