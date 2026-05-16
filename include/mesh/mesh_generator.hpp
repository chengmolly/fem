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

} // namespace fem