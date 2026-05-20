#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "../mesh/mesh.hpp"

namespace fem
{
    class QuadratureRule
    {
    public:
        virtual ~QuadratureRule() = default;
        virtual size_t num_points() const = 0;
        virtual const std::vector<double> &weights() const = 0;
        virtual const std::vector<std::vector<double>> &points() const = 0;

        template <typename Func>
        double integrate(Func &&func) const
        {
            double res = 0.0;
            const auto &pts = points();
            const auto &wts = weights();
            for (size_t i = 0; i < num_points(); ++i)
            {
                res += wts[i] * func(pts[i]);
            }
            return res;
        }

    protected:
        QuadratureRule() = default;
    };

    class GaussLegendre1D : public QuadratureRule
    {
    public:
        explicit GaussLegendre1D(size_t n) : _n(n)
        {
            compute_points_and_weights();
        }
        size_t num_points() const override { return _n; }
        const std::vector<double> &weights() const override { return _wts; }
        const std::vector<std::vector<double>> &points() const override { return _pts; }

    private:
        void compute_points_and_weights()
        {
            static const double gauss_points_1d[5][5] =
                {
                    {0.0, 0.0, 1.0},
                    {-0.5773502691896257, 0.5773502691896257, 0.0},
                    {-0.7745966692414834, 0.0, 0.7745966692414834, 0.0},
                    {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526, 0.0},
                    {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640}};

            static const double gauss_weights_1d[5][5] =
                {
                    {2.0, 0.0, 0.0},
                    {1.0, 1.0, 0.0},
                    {0.5555555555555556, 0.8888888888888888, 0.5555555555555556},
                    {0.3478548451374540, 0.6521451548625459, 0.6521451548625459, 0.3478548451374540},
                    {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891}};

            if (_n > 5)
            {
                throw std::invalid_argument("GaussLegendre1D: n must be 1, 2, 3, 4, or 5");
            }

            _pts.resize(_n);
            _wts.resize(_n);

            for (size_t i = 0; i < _n; ++i)
            {
                _pts[i] = {gauss_points_1d[_n - 1][i]};
                _wts[i] = gauss_weights_1d[_n - 1][i];
            }
        }
        size_t _n;
        std::vector<double> _wts;
        std::vector<std::vector<double>> _pts;
    };

    class GaussLegendre2D_Tri : public QuadratureRule
    {
        explicit GaussLegendre2D_Tri(size_t n) : _n(n)
        {
            compute_points_and_weights();
        }

        size_t num_points() const override { return _n; }
        const std::vector<double> &weights() const override { return _wts; }
        const std::vector<std::vector<double>> &points() const override { return _pts; }

    private:
        void compute_points_and_weights()
        {
            if (_n == 1)
            {
                _pts = {{1.0 / 3.0, 1.0 / 3.0}};
                _wts = {0.5};
            }
            else if (_n == 3)
            {
                _pts = {{0.5, 0.5}, {0.5, 0.0}, {0.0, 0.5}};
                _wts = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};
            }
            else if (_n == 4)
            {
                _pts = {{1.0 / 3.0, 1.0 / 3.0}, {0.2, 0.6}, {0.6, 0.2}, {0.2, 0.2}};
                _wts = {-0.5625, 0.5208333333333334, 0.5208333333333334, 0.5208333333333334};
            }
            else
            {
                throw std::invalid_argument("GaussLegendre2D_Tri: n must be 1, 3, or 4");
            }
        }
        size_t _n;
        std::vector<double> _wts;
        std::vector<std::vector<double>> _pts;
    };

    class GaussLegendre2D_Quad : public QuadratureRule
    {
    public:
        explicit GaussLegendre2D_Quad(size_t n) : _n(n)
        {
            GaussLegendre1D gauss1d(_n);
            auto pts1d = gauss1d.points();
            auto wts1d = gauss1d.weights();

            _pts.reserve(_n * _n);
            _wts.reserve(_n * _n);

            for (size_t i = 0; i < _n; ++i)
            {
                for (size_t j = 0; j < _n; ++j)
                {
                    _wts.push_back(wts1d[i] * wts1d[j]);
                    _pts.push_back({pts1d[i][0], pts1d[j][0]});
                }
            }
        }

        size_t num_points() const { return _n * _n; }
        const std::vector<double> &weights() const override { return _wts; }
        const std::vector<std::vector<double>> &points() const override { return _pts; }

    private:
        size_t _n;
        std::vector<double> _wts;
        std::vector<std::vector<double>> _pts;
    };

    class GaussLegendre3D_Hex : public QuadratureRule
    {
    public:
        explicit GaussLegendre3D_Hex(size_t n) : _n(n)
        {
            GaussLegendre1D gauss1d(_n);
            auto pts1d = gauss1d.points();
            auto wts1d = gauss1d.weights();

            _pts.reserve(_n * _n * _n);
            _wts.reserve(_n * _n * _n);

            for (size_t i = 0; i < _n; ++i)
            {
                for (size_t j = 0; j < _n; ++j)
                {
                    for (size_t k = 0; k < _n; ++k)
                    {
                        _wts.push_back(wts1d[i] * wts1d[j] * wts1d[k]);
                        _pts.push_back({pts1d[i][0], pts1d[j][0], pts1d[k][0]});
                    }
                }
            }
        }

        size_t num_points() const override { return _n * _n * _n; }
        const std::vector<double> &weights() const override { return _wts; }
        const std::vector<std::vector<double>> &points() const override { return _pts; }

    private:
        size_t _n;
        std::vector<double> _wts;
        std::vector<std::vector<double>> _pts;
    };

    class GaussLegendre3D_Tet : public QuadratureRule
    {
    public:
        explicit GaussLegendre3D_Tet(size_t n) : _n(n)
        {
            compute_points_and_weights();
        }
        size_t num_points() const override { return _n; }
        const std::vector<double> &weights() const override { return _wts; }
        const std::vector<std::vector<double>> &points() const override { return _pts; }

    private:
        void compute_points_and_weights()
        {
            if (_n == 1)
            {
                _pts = {{0.25, 0.25, 0.25}};
                _wts = {1.0 / 6.0};
            }
            else if (_n == 4)
            {
                _pts = {{0.5854101966249685, 0.13819660112500105, 0.13819660112500105},
                        {0.1381966011250105, 0.5854101966249685, 0.13819660112500105},
                        {0.1381966011250105, 0.1381966011250105, 0.5854101966249685},
                        {0.5854101966249685, 0.1381966011250105, 0.1381966011250105}};
                _wts = {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0};
            }
            else
            {
                throw std::invalid_argument("GaussLegendre3D_Tet: n must be 1 or 4");
            }
        }
        size_t _n;
        std::vector<double> _wts;
        std::vector<std::vector<double>> _pts;
    };

    class QuadratureFactory
    {
    public:
        static std::unique_ptr<QuadratureRule> create(ElementType type, size_t n)
        {
            switch (type)
            {
            case ElementType::LINE2:
            case ElementType::LINE3:
                return std::make_unique<GaussLegendre1D>(n);
            case ElementType::TRI3:
                return std::make_unique<GaussLegendre2D_Tri>(n);
            case ElementType::QUAD4:
            case ElementType::QUAD8:
                return std::make_unique<GaussLegendre2D_Quad>(n);
            case ElementType::TET4:
                return std::make_unique<GaussLegendre3D_Tet>(n);
            case ElementType::HEX8:
                return std::make_unique<GaussLegendre3D_Hex>(n);
            default:
                throw std::invalid_argument("QuadratureFactory: unknown element type");
            }
        }
    };

} // namespace fem