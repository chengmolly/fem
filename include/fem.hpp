#pragma once

#include "core/vector.hpp"
#include "core/matrix.hpp"
#include "core/sparse_matrix.hpp"

#include "mesh/mesh.hpp"
#include "mesh/mesh_generator.hpp"

#include "fem/quadrature.hpp"
#include "fem/shape_function.hpp"
#include "fem/element_matrix.hpp"
#include "fem/assembler.hpp"
#include "fem/fem_solver.hpp"

#include "solver/linear_solver.hpp"
#include "bc/boundary_condition.hpp"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

#define FEM_VERSION_MAJOR 1
#define FEM_VERSION_MINOR 0
#define FEM_VERSION_PATCH 0

namespace fem
{
    constexpr int version_major = FEM_VERSION_MAJOR;
    constexpr int version_minor = FEM_VERSION_MINOR;
    constexpr int version_patch = FEM_VERSION_PATCH;

    inline const std::string version_fem()
    {
        return "FEM v" +
               std::to_string(version_major) + "." +
               std::to_string(version_minor) + "." +
               std::to_string(version_patch);
    }
} // naemspace fem
