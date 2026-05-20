#pragma once

#include "../mesh/mesh.hpp"
#include "./quadrature.hpp"

namespace fem
{
    class ShapeFunction
    {
    public:
        virtual ~ShapeFunction() = default;

        virtual size_t num_shape_functions() const = 0;
        virtual ElementType element_type() const = 0;

        virtual double shape_value(size_t i, const std::vector<double> &points_xi) const = 0;
        virtual std::vector<double> shape_grad(size_t i, const std::vector<double> &points_xi) const = 0;

        virtual std::vector<double> shape_values(const std::vector<double> &points_xi) const = 0;
        virtual std::vector<std::vector<double>> shape_grads(const std::vector<double> &points_xi) const = 0;
    };

    class ShapeFunctionLine : public ShapeFunction
    {
    public:
        ShapeFunctionLine(size_t n) : n_() {}

        size_t num_shape_functions() const override { return n_; }

        ElementType element_type() const override
        {
            return n_ == 2 ? ElementType::LINE2 : ElementType::LINE3;
        }

        double shape_value(size_t i, const std::vector<double> &points_xi) const override
        {
            double x = points_xi[0];
            if (n_ == 2)
            {
                if (i == 0)
                    return 0.5 * (1.0 - x);
                if (i == 1)
                    return 0.5 * (1.0 + x);
            }
            else if (n_ == 3)
            {
                if (i == 0)
                    return 0.5 * (1.0 - x) * (1.0 - 2.0 * x);
                if (i == 1)
                    return 0.5 * (1.0 + x) * (1.0 - x);
                if (i == 2)
                    return 0.5 * (1.0 + x) * (1.0 + 2.0 * x);
            }
            else
            {
                throw std::runtime_error("Invalid number of shape functions");
            }
            return 0.0;
        }

        std::vector<double> shape_grad(size_t i, const std::vector<double> &points_xi) const override
        {
            double x = points_xi[0];
            std::vector<double> grad(1, 0.0);

            if (n_ == 2)
            {
                if (i == 0)
                    grad[0] = -0.5;
                if (i == 1)
                    grad[0] = 0.5;
            }
            else if (n_ == 3)
            {
                if (i == 0)
                    grad[0] = -0.5 * (-2.0 * x + 1.0);
                if (i == 1)
                    grad[0] = -0.5 * (1.0 - 2.0 * x);
                if (i == 2)
                    grad[0] = 0.5 * (1.0 + 2.0 * x);
            }
            else
            {
                throw std::runtime_error("Invalid number of shape functions");
            }
            return grad;
        }

        std::vector<double> shape_values(const std::vector<double> &points_xi) const override
        {
            std::vector<double> values(num_shape_functions());
            for (size_t i = 0; i < num_shape_functions(); ++i)
            {
                values[i] = shape_value(i, points_xi);
            }
            return values;
        }

        std::vector<std::vector<double>> shape_grads(const std::vector<double> &points_xi) const override
        {
            std::vector<std::vector<double>> d_values(num_shape_functions());
            for (size_t i = 0; i < num_shape_functions(); ++i)
            {
                d_values[i] = shape_grad(i, points_xi);
            }
            return d_values;
        }

    private:
        size_t n_;
    };

    class ShapeFunctionTri : public ShapeFunction
    {
    public:
        ShapeFunctionTri(size_t n) : n_(n) {}
        size_t num_shape_functions() const override { return n_; }

        ElementType element_type() const override
        {
            if (n_ == 3)
                return ElementType::TRI3;
        }

        double shape_value(size_t i, const std::vector<double> &points_xi) const override
        {
            double x = points_xi[0], y = points_xi[1];
            if (n_ == 3)
            {
                if (i == 0)
                    return 1.0 - x - y;
                if (i == 1)
                    return x;
                if (i == 2)
                    return y;
            }
            else if (n_ == 6)
            {
                if (i == 0)
                    return 2.0 * (1.0 - x - y) * (1.0 - 2.0 * x - 2.0 * y);
                if (i == 1)
                    return 2.0 * x * (1.0 - 2.0 * x - 2.0 * y);
                if (i == 2)
                    return 2.0 * y * (1.0 - 2.0 * x - 2.0 * y);
                if (i == 3)
                    return 4.0 * x * (1.0 - x - y);
                if (i == 4)
                    return 4.0 * y * (1.0 - x - y);
                if (i == 5)
                    return 4.0 * x * y;
            }
            else
            {
                throw std::runtime_error("Invalid number of shape functions");
            }
            return 0.0;
        }

        std::vector<double> shape_grad(size_t i, const std::vector<double> &points_xi) const override
        {
            double x = points_xi[0], y = points_xi[1];
            std::vector<double> grad(2, 0.0);
            if (n_ == 3)
            {
                if (i == 0)
                {
                    grad[0] = -1.0;
                    grad[1] = -1.0;
                }
                if (i == 1)
                {
                    grad[0] = 1.0;
                    grad[1] = 0.0;
                }
                if (i == 2)
                {
                    grad[0] = 0.0;
                    grad[1] = 1.0;
                }
            }
            else if (n_ == 6)
            {
                if (i == 0)
                {
                    grad[0] = -2.0 * (1.0 - 2.0 * y);
                    grad[1] = -2.0 * (1.0 - 2.0 * x);
                }
                if (i == 1)
                {
                    grad[0] = -2.0 * (1.0 - 2.0 * y);
                    grad[1] = 2.0 * (1.0 - 2.0 * x);
                }
                if (i == 2)
                {
                    grad[0] = 2.0 * (1.0 - 2.0 * y);
                    grad[1] = 2.0 * (1.0 - 2.0 * x);
                }
                if (i == 3)
                {
                    grad[0] = 4.0 * (1.0 - y);
                    grad[1] = 2.0 * (1.0 - 2.0 * x);
                }
                if (i == 4)
                {
                    grad[0] = 4.0 * (1.0 - y);
                    grad[1] = 4.0 * (1.0 - x);
                }
                if (i == 5)
                {
                    grad[0] = -4.0 * (1.0 - y);
                    grad[1] = 4.0 * (1.0 - x);
                }
            }
            else
            {
                throw std::runtime_error("Invalid number of shape functions");
            }
            return grad;
        }

        std::vector<double> shape_values(const std::vector<double> &points_xi) const override
        {
            std::vector<double> values(num_shape_functions());
            for (size_t i; i < num_shape_functions(); ++i)
            {
                values[i] = shape_value(i, points_xi);
            }
            return values;
        }

        std::vector<std::vector<double>> shape_grads(const std::vector<double> &points_xi) const override
        {
            std::vector<std::vector<double>> d_values(num_shape_functions());
            for (size_t i = 0; i < num_shape_functions(); ++i)
            {
                d_values[i] = shape_grad(i, points_xi);
            }
            return d_values;
        }

    private:
        size_t n_;
    };

    class ShapeFunctionQuad : public ShapeFunction
    {
    public:
        ShapeFunctionQuad(size_t n) : n_(n) {}
        size_t num_shape_functions() const override { return n_; }

        ElementType element_type() const override
        {
            if (n_ == 4)
                return ElementType::QUAD4;
        }

        double shape_value(size_t i, const std::vector<double> &point_xi) const override
        {
            double x = point_xi[0], y = point_xi[1];
            if (n_ == 4)
            {
                if (i == 0)
                    return (1 - x) * (1 - y);
                if (i == 1)
                    return (1 + x) * (1 - y);
                if (i == 2)
                    return (1 + x) * (1 + y);
                if (i == 3)
                    return (1 - x) * (1 + y);
            }
            return 0.0;
        }
        std::vector<double> shape_grad(size_t i, const std::vector<double> &points_xi) const override
        {
            double x = points_xi[0], y = points_xi[1];
            std::vector<double> grad(2, 0.0);
            if (n_ == 4)
            {
                if (i == 0)
                {
                    grad[0] = -1.0;
                    grad[1] = -1.0;
                }
                if (i == 1)
                {
                    grad[0] = 1.0;
                    grad[1] = -1.0;
                }
                if (i == 2)
                {
                    grad[0] = 1.0;
                    grad[1] = 1.0;
                }
                if (i == 3)
                {
                    grad[0] = -1.0;
                    grad[1] = 1.0;
                }
            }
            return grad;
        }
        std::vector<double> shape_values(const std::vector<double> &points_xi) const override
        {
            std::vector<double> values(num_shape_functions());
            for (size_t i; i < num_shape_functions(); ++i)
            {
                values[i] = shape_value(i, points_xi);
            }
            return values;
        }

        std::vector<std::vector<double>> shape_grads(const std::vector<double> &points_xi) const override
        {
            std::vector<std::vector<double>> d_values(num_shape_functions());
            for (size_t i = 0; i < num_shape_functions(); ++i)
            {
                d_values[i] = shape_grad(i, points_xi);
            }
            return d_values;
        }

    private:
        size_t n_;
    };

    class ShapeFunctionTet : public ShapeFunction
    {
    public:
        ShapeFunctionTet(size_t n) : n_(n) {}

        size_t num_shape_functions() const override { return n_; }

        ElementType element_type() const override
        {
            if (n_ == 4)
                return ElementType::TET4;
        }

        double shape_value(size_t i, const std::vector<double> &points_xi) const override
        {
            double x = points_xi[0], y = points_xi[1], z = points_xi[2];
            if (n_ == 4)
            {
                if (i == 0)
                    return (1 - x - y - z);
                if (i == 1)
                    return (x);
                if (i == 2)
                    return (y);
                if (i == 3)
                    return (z);
            }
            return 0.0;
        }

        std::vector<double> shape_grad(size_t i, const std::vector<double> &points_xi) const override
        {
            std::vector<double> grad(3, 0.0);
            if (n_ == 4)
            {
                if (i == 0)
                {
                    grad[0] = -1.0;
                    grad[1] = -1.0;
                    grad[2] = -1.0;
                }
                if (i == 1)
                {
                    grad[0] = 1.0;
                    grad[1] = 0.0;
                    grad[2] = 0.0;
                }
                if (i == 2)
                {
                    grad[0] = 0.0;
                    grad[1] = 1.0;
                    grad[2] = 0.0;
                }
                if (i == 3)
                {
                    grad[0] = 0.0;
                    grad[1] = 0.0;
                    grad[2] = 1.0;
                }
            }
            return grad;
        }
        std::vector<double> shape_values(const std::vector<double> &points_xi) const override
        {
            std::vector<double> values(num_shape_functions());
            for (size_t i; i < num_shape_functions(); ++i)
            {
                values[i] = shape_value(i, points_xi);
            }
            return values;
        }

        std::vector<std::vector<double>> shape_grads(const std::vector<double> &points_xi) const override
        {
            std::vector<std::vector<double>> d_values(num_shape_functions());
            for (size_t i = 0; i < num_shape_functions(); ++i)
            {
                d_values[i] = shape_grad(i, points_xi);
            }
            return d_values;
        }

    private:
        size_t n_;
    };

    class ShapeFunctionHex : public ShapeFunction
    {
    public:
        ShapeFunctionHex(size_t n) : n_(n) {}

        size_t num_shape_functions() const override { return n_; }
        ElementType element_type() const override
        {
            if (n_ == 8)
                return ElementType::HEX8;
        }

        double shape_value(size_t i, const std::vector<double> &points_xi) const override
        {
            double x = points_xi[0], y = points_xi[1], z = points_xi[2];
            if (n_ == 8)
            {
                if (i == 0)
                    return (1 - x) * (1 - y) * (1 - z);
                if (i == 1)
                    return (1 + x) * (1 - y) * (1 - z);
                if (i == 2)
                    return (1 + x) * (1 + y) * (1 - z);
                if (i == 3)
                    return (1 - x) * (1 + y) * (1 - z);
                if (i == 4)
                    return (1 - x) * (1 - y) * (1 + z);
                if (i == 5)
                    return (1 + x) * (1 - y) * (1 + z);
                if (i == 6)
                    return (1 + x) * (1 + y) * (1 + z);
                if (i == 7)
                    return (1 - x) * (1 + y) * (1 + z);
            }
            return 0.0;
        }
        std::vector<double> shape_grad(size_t i, const std::vector<double> &points_xi) const override
        {
            std::vector<double> grad(3, 0.0);
            if (n_ == 8)
            {
                if (i == 0)
                {
                    grad[0] = -1.0;
                    grad[1] = -1.0;
                    grad[2] = -1.0;
                }
                if (i == 1)
                {
                    grad[0] = 1.0;
                    grad[1] = -1.0;
                    grad[2] = -1.0;
                }
                if (i == 2)
                {
                    grad[0] = 1.0;
                    grad[1] = 1.0;
                    grad[2] = -1.0;
                }
                if (i == 3)
                {
                    grad[0] = -1.0;
                    grad[1] = 1.0;
                    grad[2] = -1.0;
                }
                if (i == 4)
                {
                    grad[0] = -1.0;
                    grad[1] = -1.0;
                    grad[2] = 1.0;
                }
                if (i == 5)
                {
                    grad[0] = 1.0;
                    grad[1] = -1.0;
                    grad[2] = 1.0;
                }
                if (i == 6)
                {
                    grad[0] = 1.0;
                    grad[1] = 1.0;
                    grad[2] = 1.0;
                }
                if (i == 7)
                {
                    grad[0] = -1.0;
                    grad[1] = 1.0;
                    grad[2] = 1.0;
                }
            }
            return grad;
        }

        std::vector<double> shape_values(const std::vector<double> &points_xi) const override
        {
            std::vector<double> values(num_shape_functions());
            for (size_t i; i < num_shape_functions(); ++i)
            {
                values[i] = shape_value(i, points_xi);
            }
            return values;
        }

        std::vector<std::vector<double>> shape_grads(const std::vector<double> &points_xi) const override
        {
            std::vector<std::vector<double>> d_values(num_shape_functions());
            for (size_t i = 0; i < num_shape_functions(); ++i)
            {
                d_values[i] = shape_grad(i, points_xi);
            }
            return d_values;
        }

    private:
        size_t n_;
    };

    class ShapeFunctionFactory
    {
    public:
        static std::unique_ptr<ShapeFunction> create(ElementType type)
        {
            size_t order = ElementInfo::get_order(type);
            switch (type)
            {
            case ElementType::LINE2:
            case ElementType::LINE3:
                return std::make_unique<ShapeFunctionLine>(order);
            case ElementType::TRI3:
                return std::make_unique<ShapeFunctionTri>(order);
            case ElementType::QUAD4:
            case ElementType::QUAD8:
                return std::make_unique<ShapeFunctionQuad>(order);
            case ElementType::TET4:
                return std::make_unique<ShapeFunctionTet>(order);
            case ElementType::HEX8:
                return std::make_unique<ShapeFunctionHex>(order);
            default:
                throw std::runtime_error("Unknown element type");
            }
        }
        static std::unique_ptr<ShapeFunction> create(size_t order, int dim)
        {
            if (dim == 1)
                return std::make_unique<ShapeFunctionLine>(order);
            if (dim == 2)
                return std::make_unique<ShapeFunctionQuad>(order);
            if (dim == 3)
                return std::make_unique<ShapeFunctionHex>(order);
            throw std::runtime_error("Unsupported dimension");
        }
    };

} // namespace fem