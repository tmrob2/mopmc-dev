set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/Geometry.cpp"
        "${dir}/Geometry.h"
        "${dir}/IterativeSolver.h"
        "${dir}/IterativeSolver.cpp"
        "${dir}/InducedEquationSolver.h"
        "${dir}/InducedEquationSolver.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)