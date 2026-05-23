#pragma once

#include <memory>
#include <string>
#include <vector>
#include "../mesh/mesh.hpp"
#include "../core/sparse_matrix.hpp"
#include "assembler.hpp"
#include "quadrature.hpp"
#include "shape_function.hpp"
#include "../solver/linear_solver.hpp"
#include "../bc/boundary_condition.hpp"
#include "../utils/timer.hpp"
#include "../utils/logger.hpp"

namespace fem
{
    struct FEMConfig
    {
        // Solver opti
        std::string solver_type = "CG";
        std::string Preconditioner = "Jacobi";

        // iteration opti
        size_t max_iterations = 1000;
        double tolerance = 1e-10;
        size_t restart = 50;

        // mesh opti
        size_t quad_order = 2;
        bool use_threads = false;
        size_t num_threads = 1;

        // output opti
        bool verbose = true;
        bool save_result = false;
        std::string result_path = "./result.vtk";

        // phy para
        double diffusion_coef = 1.0;
        double source_term = 0.0;
    };

    class FEMSolver
    {
    public:
        FEMSolver() : config_(), is_initialized_(false) {}

        explicit FEMSolver(const FEMConfig &config) : config_(config), is_initialized_(false) {}

        void configure(const FEMConfig &config) { config_ = config; }
        void initialize(Mesh &mesh)
        {
            mesh_ = &mesh;
            pde_data_.problem_type = ProblemType::Poisson;
            pde_data_.num_equs = 1;

            if (config_.diffusion_coef != 1.0)
            {
                pde_data_.diffusion_coef = [this](const std::vector<double> &)
                {
                    return config_.diffusion_coef;
                };
            }

            if (config_.source_term != 0.0)
            {
                pde_data_.source_term = [this](const std::vector<double> &)
                {
                    return config_.source_term;
                };
            }

            assembler_.init(mesh, pde_data_, config_.quad_order);

            init_solver();

            is_initialized_ = true;

            TimerManager::instance().stop("Initialization");

            LOG_INFO(" FEM Solver initialized SUCCESSFULLY!");
            LOG_INFO(" Nodes:{}", mesh.num_nodes());
            LOG_INFO(" Elements:{}", mesh.num_elements());
            LOG_INFO(" Solver:{}", config_.solver_type);
            LOG_INFO(" Preconditioner:{}", config_.Preconditioner);
        }

        void set_boundary_conditions(const BCManager &bc)
        {
            bc_manager_ = bc;
        }

        void set_dirichlet_bc(const std::vector<size_t> &nodes, const std::vector<double> &values)
        {
            bc_manager_.add_dirichlet(nodes, values);
        }

        void set_neumann_bc(const std::vector<size_t> &nodes, const std::vector<double> &values)
        {
            bc_manager_.add_neumann(nodes, values);
        }

        std::vector<double> solve()
        {
            if (!is_initialized_)
            {
                throw std::runtime_error("FEM Solver is not initialized");
            }

            TimerManager::instance().start("Assembly");
            LOG_INFO("Starting matrix assembly...");

            assembler_.assemble(global_matrix_, global_rhs_);

            TimerManager::instance().stop("Assembly");
            LOG_INFO("Matrix assembly completed.NNZ:{}", global_matrix_.values());

            TimerManager::instance().start("Boundary Conditions");
            apply_boundary_conditions();
            TimerManager::instance().stop("Boundary Conditions");

            TimerManager::instance().start("Solver");
            LOG_INFO("Starting solver...");

            std::vector<double> solution;
            solver_->solve(global_matrix_, global_rhs_, solution);

            TimerManager::instance().stop("Solver");

            SolverResult result = solver_->get_result();
            LOG_INFO("Solver finished with status:{}", static_cast<int>(result.status));
            LOG_INFO("Iterations:{}", result.iterations);
            LOG_INFO("Residual:{}", result.residual_norm);

            solution_ = solution;

            TimerManager::instance().start("Post Process");
            post_process();
            TimerManager::instance().stop("Post Process");

            return solution;
        }

        const std::vector<double> &solution() const { return solution_; }

        Mesh &mesh() { return *mesh_; }
        const Mesh &mesh() const { return *mesh_; }

        const SparseMatrixCRS &matrix() const { return global_matrix_; }

    private:
        void init_solver()
        {
            auto solver_type = LinearSolverFactory::Type::CG;

            if (config_.solver_type == "CG")
            {
                solver_type = LinearSolverFactory::Type::CG;
            }
            else if (config_.solver_type == "GMRES")
            {
                solver_type = LinearSolverFactory::Type::GMRES;
            }

            solver_ = LinearSolverFactory::create(solver_type);
            solver_->set_max_iterations(config_.max_iterations);
            solver_->set_tolerance(config_.tolerance);

            if (config_.Preconditioner != "None")
            {
                auto prec = LinearSolverFactory::create_preconditioner(config_.Preconditioner);
                if (prec)
                {
                    prec->build(global_matrix_);
                    solver_->set_preconditioner(prec);
                }
            }
        }

        void apply_boundary_conditions()
        {
            LOG_INFO("Applying boundary conditions...");

            bc_manager_.apply_penalty(global_matrix_, global_rhs_, 1e15);
        }

        void post_process()
        {
            if (!config_.save_result)
                return;

            LOG_INFO("Writing output to {}", config_.result_path);

            // VTK
            write_vtk(config_.result_path);
        }

        void write_vtk(const std::string &path) const
        {
            std::ofstream ofs(path);
            if (!ofs.is_open())
            {
                LOG_ERROR("Failed to open file {}", path);
                return;
            }

            ofs << "#vtk DataFile Version 1.0\n";
            ofs << "FEM Solution\n";
            ofs << "ASCII\n";
            ofs << "DATASET UNSTRUCTURED_GUID\n\n";

            ofs << "POINTS " << mesh_->num_nodes() << " double\n";
            for (size_t i = 0; i < mesh_->num_nodes(); ++i)
            {
                const auto &node = mesh_->node(i);
                ofs << node.x() << " " << node.y() << " " << node.z() << "\n";
            }
            ofs << "\n";

            size_t total_size = 0;
            for (size_t e = 0; e < mesh_->num_elements(); ++e)
            {
                total_size += mesh_->element(e).n_nodes() + 1;
            }

            ofs << "CELLS " << mesh_->num_elements() << " " << total_size << "\n";
            for (size_t e = 0; e < mesh_->num_elements(); ++e)
            {
                const auto &elem = mesh_->element(e);
                ofs << elem.n_nodes();
                for (size_t n = 0; n < elem.n_nodes(); ++n)
                {
                    ofs << " " << elem[n];
                }
                ofs << "\n";
            }
            ofs << "\n";

            ofs << "CELL_TYPES " << mesh_->num_elements() << "\n";
            for (size_t e = 0; e < mesh_->num_elements(); ++e)
            {
                size_t vtk_type = get_vtk_cell_type(mesh_->element(e).type());
                ofs << vtk_type << "\n";
            }
            ofs << "\n";

            ofs << "POINT_DATA " << mesh_->num_nodes() << "\n";
            ofs << "SCALARS solution double 1\n";
            ofs << "LOOKUP_TABLE default\n";
            for (size_t i = 0; i < solution_.size(); ++i)
            {
                ofs << solution_[i] << "\n";
            }

            ofs.close();
            LOG_INFO("Output written SUCCESSFULLY!");
        }

        static size_t get_vtk_cell_type(ElementType type)
        {
            switch (type)
            {
            case ElementType::LINE2:
                return 3;
            case ElementType::TRI3:
                return 5;
            case ElementType::QUAD4:
                return 9;
            case ElementType::TET4:
                return 10;
            case ElementType::HEX8:
                return 12;
            default:
                return 5;
            }
        }

        FEMConfig config_;
        bool is_initialized_;

        Mesh *mesh_;
        PDEData pde_data_;

        Assembler assembler_;
        BCManager bc_manager_;

        std::unique_ptr<LinearSolver> solver_;

        SparseMatrixCRS global_matrix_;
        std::vector<double> global_rhs_;
        std::vector<double> solution_;
    };

    template <typename MeshType, typename BCFunc>
    std::vector<double> solve_fem(MeshType &mesh, BCFunc &&bc_func)
    {
        FEMConfig config;
        FEMSolver solver(config);
        solver.initialize(mesh);
        bc_func(solver);
        return solver.solve();
    }
} // namespace fem
