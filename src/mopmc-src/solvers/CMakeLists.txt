set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/Geometry.cpp"
        "${dir}/Geometry.h"
        "${dir}/ValueIteration.h"
        "${dir}/ValueIteration.cpp"
        "${dir}/InducedEquationSolver.h"
        "${dir}/InducedEquationSolver.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)