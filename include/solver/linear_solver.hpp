#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <limits>
#include "../core/sparse_matrix.hpp"

namespace fem
{
    enum class SolverStatus
    {
        Converged,
        Diverged,
        MaxIterations,
        Breakdown,
        NotConverged,
        Error
    };

    struct SolverResult
    {
        SolverStatus status;
        size_t iterations;
        double residual_norm;
        double error;
        std::string message;

        SolverResult() : status(SolverStatus::Error), iterations(0),
                         residual_norm(0.0), error(0.0), message("") {}

        bool converged() const
        {
            return status == SolverStatus::Converged;
        }
    };

    class LinearSolver
    {
    public:
        virtual ~LinearSolver() = default;
        virtual void set_max_iterations(size_t max_iterations) = 0;
        virtual void set_tolerance(double tol) = 0;
        virtual void set_preconditioner(std::shared_ptr<Preconditioner> prec) = 0;
        virtual void solve(const SparseMatrixCRS &A, const std::vector<double> &b, std::vector<double> &x) = 0;

        virtual SolverResult get_result() const = 0;

    protected:
        LinearSolver() = default;
    };

    class Preconditioner
    {
    public:
        virtual ~Preconditioner() = default;

        virtual void build(const SparseMatrixCRS &A) = 0;
        virtual void apply(const std::vector<double> &r, std::vector<double> &z) const = 0;
        virtual std::string name() const = 0;
    };

    class JacobiPreconditioner : public Preconditioner
    {
    public:
        JacobiPreconditioner() : diag_() {}

        void build(const SparseMatrixCRS &A) override
        {
            diag_ = A.diagonal();
            for (auto &d : diag_)
            {
                if (std::abs(d) > 1e-14)
                {
                    d = 1.0 / d;
                }
                else
                {
                    d = 1.0;
                }
            }
        }

        void apply(const std::vector<double> &r, std::vector<double> &z) const override
        {
            z.resize(r.size());
            for (size_t i = 0; i < r.size(); ++i)
            {
                z[i] = diag_[i] * r[i];
            }
        }

        std::string name() const override { return "Jacobi"; }

    private:
        std::vector<double> diag_;
    };

    class SSORPreconditioner : public Preconditioner
    {
    public:
        explicit SSORPreconditioner(double omega = 1.0) : omega_(omega), diag_() {}

        void build(const SparseMatrixCRS &A) override
        {
            diag_ = A.diagonal();
            for (auto &d : diag_)
            {
                if (std::abs(d) > 1e-14)
                {
                    d = 1.0 / d;
                }
                else
                {
                    d = 1.0;
                }
            }
        }

        void apply(const std::vector<double> &r, std::vector<double> &z) const override
        {
            z.resize(r.size());
            const auto &row_ptr = diag_.row_ptr();
            const auto &col_idx = diag_.col_idx();
            const auto &values = diag_.values();

            for (size_t i = 0; i < r.size(); ++i)
            {
                double sum = 0.0;
                for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
                {
                    int col = col_idx[j];
                    if (col < static_cast<int>(i))
                    {
                        sum += values[j] * z[col];
                    }
                }
                z[i] = omega_ * (diag_[i] * (r[i] - sum)) + (1.0 - omega_) * z[i];
            }

            for (size_t i = r.size(); i-- > 0;)
            {
                double sum = 0.0;
                for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
                {
                    int col = col_idx[j];
                    if (col > static_cast<int>(i))
                    {
                        sum += values[j] * z[col];
                    }
                }
                z[i] = omega_ * (diag_[i] * (r[i] - sum)) + (1.0 - omega_) * z[i];
            }
        }
        std::string name() const override { return "SSOR"; }

    private:
        double omega_;
        SparseMatrixCRS diag_;
    };

    class ILU0Preconditioner : public Preconditioner
    {
    };

    class CGSolver : public LinearSolver
    {
    public:
        CGSolver() : max_iter_(1000), tol_(1e-10), prec_(nullptr), result_() {}

        void set_max_iterations(size_t max_iter) override
        {
            max_iter_ = max_iter;
        }

        void set_tolerance(double tol) override
        {
            tol_ = tol;
        }

        void set_preconditioner(std::shared_ptr<Preconditioner> prec) override
        {
            prec_ = prec;
        }

        void solve(const SparseMatrixCRS &A, const std::vector<double> &b, std::vector<double> &x)
        {
            if (x.size() != b.size())
                x.assign(b.size(), 0.0);

            std::vector<double> r(b.size()), p(b.size()), Ap(b.size()), z(b.size());

            A.mv(x, r);
            for (size_t i = 0; i < b.size(); ++i)
            {
                r[i] = b[i] - r[i];
            }

            double r_norm = vector_norm(r);
            double initial_norm = r_norm;

            size_t iter = 0;
            result_.iterations = 0;
            result_.residual_norm = r_norm;

            while (iter < max_iter_ && r_norm > tol_ * initial_norm)
            {
                if (prec_)
                {
                    prec_->apply(r, z);
                }
                else
                {
                    z = r;
                }

                double rz_new = vector_dot(r, z);
                if (iter == 0)
                {
                    p = z;
                }
                else
                {
                    double rz_old = result_.rz_old_;
                    double beta = rz_new / rz_old;
                    for (size_t i = 0; i < b.size(); ++i)
                    {
                        p[i] = z[i] + beta * p[i];
                    }
                }

                A.mv(p, Ap);

                double pAp = vector_dot(p, Ap);
                if (std::abs(pAp) < 1e-14)
                {
                    result_.status = SolverStatus::Breakdown;
                    break;
                }

                double alpha = rz_new / pAp;
                for (size_t i = 0; i < b.size(); ++i)
                {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * Ap[i];
                }

                r_norm = vector_norm(r);
                result_.iterations = iter + 1;
                result_.residual_norm = r_norm;
                result_.rz_old_ = rz_new;
            }

            if (result_.iterations >= max_iter_)
            {
                result_.status = SolverStatus::MaxIterations;
            }
            else if (result_.residual_norm <= tol_ * initial_norm)
            {
                result_.status = SolverStatus::Converged;
            }
        }
        SolverResult get_result() const override { return result_; }

    private:
        size_t max_iter_;
        double tol_;
        std::shared_ptr<Preconditioner> prec_;
        SolverResult result_;

        static double vector_norm(const std::vector<double> &v)
        {
            double sum = 0.0;
            for (double x : v)
                sum += x * x;
            return std::sqrt(sum);
        }

        static double vector_dot(const std::vector<double> &a, const std::vector<double> &b)
        {
            double sum = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }
    };

    class GMRESSolver : public LinearSolver
    {
    public:
        explicit GMRESSolver(size_t restart = 50) : restart_(restart), max_iter_(1000), tol_(1e-10), prec_(nullptr), result_() {}

        void set_max_iterations(size_t max_iter) override
        {
            max_iter_ = max_iter;
        }

        void set_tolerance(double tol) override
        {
            tol_ = tol;
        }

        void set_preconditioner(std::shared_ptr<Preconditioner> prec) override
        {
            prec_ = prec;
        }

        void solve(const SparseMatrixCRS &A, const std::vector<double> &b, std::vector<double> &x) override
        {
            if (x.size() != b.size())
                x.assign(b.size(), 0.0);
            std::vector<double> r(b.size()), w(b.size());
            std::vector<std::vector<double>> v(restart_ + 1);

            A.mv(x, r);
            for (size_t i = 0; i < b.size(); ++i)
            {
                r[i] = b[i] - r[i];
            }

            double beta = vector_norm(r);
            double initial_norm = beta;

            size_t iter = 0;
            size_t total_iter = 0;

            while (total_iter < max_iter_ && beta > tol_ * initial_norm)
            {
                // Arnoldi
                for (size_t i = 0; i < restart_ + 1; ++i)
                    v[i].assign(b.size(), 0.0);

                for (size_t i = 0; i < b.size(); ++i)
                    v[0][i] = r[i] / beta;

                std::vector<double> g(restart_ + 1, 0.0);
                g[0] = beta;

                std::vector<std::vector<double>> H(restart_ + 1, std::vector<double>(restart_, 0.0));

                bool converged = false;
                for (size_t j = 0; j < restart_ && total_iter + j < max_iter_; ++j)
                {
                    A.mv(v[j], w);

                    if (prec_)
                    {
                        std::vector<double> z(b.size());
                        prec_->apply(w, z);
                        w = z;
                    }

                    for (size_t i = 0; i <= j; ++i)
                    {
                        H[i][j] = vector_dot(v[i], w);
                        for (size_t k = 0; k < b.size(); ++k)
                        {
                            w[k] -= H[i][j] * v[i][k];
                        }
                    }
                    H[j + 1][j] = vector_norm(w);

                    if (H[j + 1][j] < 1e-14)
                    {
                        converged = true;
                        break;
                    }

                    for (size_t k = 0; k < b.size(); ++k)
                    {
                        v[j + 1][k] = w[k] / H[j + 1][j];
                    }

                    std::vector<double> yy = g;

                    for (size_t i = 0; i <= j; ++i)
                    {
                        double sum = 0.0;
                        for (size_t k = i + 111; k <= j; ++k)
                        {
                            sum += H[i][k] * yy[k];
                        }
                        yy[i] = (g[i] - sum) / H[i][j];
                    }

                    double residual = 0.0;
                    for (size_t i = j + 1; i < j + 1; ++i)
                    {
                        double sum = 0.0;
                        for (size_t k = 0; k <= j; ++k)
                        {
                            sum += H[i][k] * yy[k];
                        }
                        residual += (g[i] - sum) * (g[i] - sum);
                    }
                    beta = std::sqrt(residual);

                    if (beta <= tol_ * initial_norm)
                    {
                        converged = true;
                        break;
                    }

                    std::vector<double> yy(j + 1);

                    for (size_t i = 0; i <= j; ++i)
                    {
                        double sum = 0.0;
                        for (size_t k = i + 1; k <= j; ++k)
                        {
                            sum += H[i][k] * yy[k];
                        }
                        yy[i] = (g[i] - sum) / H[i][i];
                    }

                    for (size_t i = 0; i <= j; ++i)
                    {
                        for (size_t k = 0; k < b.size(); ++k)
                        {
                            x[k] += yy[i] * v[i][k];
                        }
                    }

                    A.mv(x, r);
                    for (size_t i = 0; i < b.size(); ++i)
                        r[i] = b[i] - r[i];
                    beta = vector_norm(r);
                    total_iter += j + 1;

                    if (converged)
                        break;
                }
            }
            result_.iterations = total_iter;
            result_.residual_norm = beta;
            result_.status = (beta <= tol_ * initial_norm) ? SolverStatus::Converged : SolverStatus::MaxIterations;
        }

        SolverResult get_result() const override { return result_; }

    private:
        size_t restart_;
        size_t max_iter_;
        double tol_;
        std::shared_ptr<Preconditioner> prec_;
        SolverResult result_;

        static double vector_norm(const std::vector<double> &v)
        {
            double sum = 0.0;
            for (double x : v)
                sum += x * x;
            return std::sqrt(sum);
        }

        static double vector_dot(const std::vector<double> &a, const std::vector<double> &b)
        {
            double sum = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
                sum += a[i] * b[i];
            return sum;
        }
    };

    class LinearSolverFactory
    {
    public:
        enum class Type
        {
            CG,
            GMRES,
            BiCGSTAB,
            LU
        };

        static std::unique_ptr<LinearSolver> create(Type type)
        {
            switch (type)
            {
            case Type::CG:
                return std::make_unique<CGSolver>();
            case Type::GMRES:
                return std::make_unique<GMRESSolver>();
            default:
                return std::make_unique<CGSolver>();
            }
        }

        static std::shared_ptr<Preconditioner> create_preconditioner(const std::string &type)
        {
            if (type == "Jacobi")
                return std::make_shared<JacobiPreconditioner>();
            if (type == "SSOR")
                return std::make_shared<SSORPreconditioner>();
            return nullptr;
        }
    };
} // namespace fem
