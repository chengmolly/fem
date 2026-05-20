#pragma once

#include "mesh.hpp"

namespace fem
{

    class MeshGenerator
    {
    public:
        virtual ~MeshGenerator() = default;
        virtual void generate(Mesh &mesh) = 0;
    };

    class Uniform1DMeshGenerator : public MeshGenerator
    {
    public:
        Uniform1DMeshGenerator(double x0, double x1, int nx) : x0_(x0), x1_(x1), nx_(nx) {}

        void generate(Mesh &mesh) override
        {
            mesh = Mesh();
            double h = (x1_ - x0_) / nx_;

            // create nodes
            for (int i = 0; i <= nx_; ++i)
            {
                double x = x0_ + i * h;
                mesh.add_node(i, {x});

                // make boundary node
                if (i == 0 || i == nx_)
                {
                    mesh.node(i).set_boundary(true);
                }
            }
            // create elements
            for (int i = 0; i < nx_; ++i)
            {
                mesh.add_element(i, ElementType::LINE2, {i, i + 1});
            }
        }

    private:
        double x0_, x1_;
        int nx_;
    };

    class Rect2DMeshGenerator : public MeshGenerator
    {
    public:
        Rect2DMeshGenerator(double x0, double y0, double x1, double y1, int nx, int ny, bool quad = true)
            : x0_(x0), y0_(y0), x1_(x1), y1_(y1), nx_(nx), ny_(ny), quad_(quad) {}

        void generate(Mesh &mesh) override
        {
            mesh = Mesh();
            double hx = (x1_ - x0_) / nx_;
            double hy = (y1_ - y0_) / ny_;

            // Create nodes
            for (int j = 0; j <= ny_; ++j)
            {
                for (int i = 0; i <= nx_; ++i)
                {
                    int id = j * (nx_ + 1) + i;
                    double x = x0_ + i * hx;
                    double y = y0_ + j * hy;
                    mesh.add_node(id, {x, y});

                    // make boundary node
                    if (i == 0 || i == nx_ || j == 0 || j == ny_)
                    {
                        mesh.node(id).set_boundary(true);
                    }
                }
            }

            // Create elements
            for (int j = 0; j < ny_; ++j)
            {
                for (int i = 0; i < nx_; ++i)
                {
                    int id = j + nx_ + i;
                    int n0 = j * (nx_ + 1) + i;
                    int n1 = n0 + 1;
                    int n2 = n0 + (nx_ + 1);
                    int n3 = n2 + 1;

                    mesh.add_element(id, ElementType::QUAD4, {n0, n1, n3, n2});
                }
            }
        }

    private:
        double x0_, y0_, x1_, y1_;
        int nx_, ny_;
        bool quad_;
    };

    class Box3DMeshGenerator : public MeshGenerator
    {
    public:
        // struct point3d
        // {
        //     double x, y, z;
        // };
        Box3DMeshGenerator(
            double x0, double y0, double z0,
            double x1, double y1, double z1,
            int nx, int ny, int nz)
            : x0_(x0), y0_(y0), z0_(z0),
              x1_(x1), y1_(y1), z1_(z1),
              nx_(nx), ny_(ny), nz_(nz) {}

        void generate(Mesh &mesh) override
        {
            mesh = Mesh();
            double hx = (x1_ - x0_) / nx_;
            double hy = (y1_ - y0_) / ny_;
            double hz = (z1_ - z0_) / nz_;

            for (int k = 0; k <= nz_; ++k)
            {
                for (int j = 0; j <= ny_; ++j)
                {
                    for (int i = 0; i <= nx_; ++i)
                    {
                        int id = k * (ny_ + 1) * (nx_ + 1) + j * (nx_ + 1) + i;
                        double x = x0_ + i * hx;
                        double y = y0_ + j * hy;
                        double z = z0_ + k * hz;
                        mesh.add_node(id, {x, y, z});

                        if (i == 0 || i == nx_ ||
                            j == 0 || j == ny_ ||
                            k == 0 || k == nz_)
                        {
                            mesh.node(id).set_boundary(true);
                        }
                    }
                }
            }

            auto node_index = [this](int i, int j, int k)
            {
                return k * (ny_ + 1) * (nx_ + 1) + j * (nx_ + 1) + i;
            };

            int elem_id = 0;
            for (int k = 0; k < nz_; ++k)
            {
                for (int j = 0; j < ny_; ++j)
                {
                    for (int i = 0; i < nx_; ++i)
                    {
                        int n0 = node_index(i, j, k);
                        int n1 = node_index(i + 1, j, k);
                        int n2 = node_index(i + 1, j + 1, k);
                        int n3 = node_index(i, j + 1, k);
                        int n4 = node_index(i, j, k + 1);
                        int n5 = node_index(i + 1, j, k + 1);
                        int n6 = node_index(i + 1, j + 1, k + 1);
                        int n7 = node_index(i, j + 1, k + 1);

                        mesh.add_element(elem_id++, ElementType::HEX8, {n0, n1, n2, n3, n4, n5, n6, n7});
                    }
                }
            }
        }

    private:
        double x0_, y0_, z0_, x1_, y1_, z1_;
        int nx_, ny_, nz_;
    };

    class MeshGeneratorFactory
    {
    public:
        enum class Type
        {
            Uniform1D,
            Rect2D,
            Box3D
        };

        static std::unique_ptr<MeshGenerator> create(Type type)
        {
            switch (type)
            {
            case Type::Uniform1D:
                return std::make_unique<Uniform1DMeshGenerator>(0.0, 1.0, 10);
            case Type::Rect2D:
                return std::make_unique<Rect2DMeshGenerator>(0.0, 0.0, 1.0, 1.0, 10, 10);
            case Type::Box3D:
                return std::make_unique<Box3DMeshGenerator>(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10, 10, 10);
            default:
                return nullptr;
            }
        }

        template <typename... Args>
        static std::unique_ptr<MeshGenerator> create(Type type, Args &&...args)
        {
            switch (type)
            {
            case Type::Uniform1D:
                return srd::make_unique<Uniform1DMeshGenerator>(std::forward<Args>(args)...);
            case Type::Rect2D:
                return std::make_unique<Rect2DMeshGenerator>(std::forward<Args>(args)...);
            case Type::Box3D:
                return std::make_unique<Box3DMeshGenerator>(std::forward<Args>(args)...);
            default:
                return nullptr;
            }
        }
    };

} // namespace fem