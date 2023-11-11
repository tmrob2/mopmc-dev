set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ConvexQuery.cpp"
        "${dir}/ConvexQuery.h"
        "${dir}/IterativeSolver.h"
        "${dir}/IterativeSolver.cpp"
        "${dir}/InducedEquationSolver.h"
        "${dir}/InducedEquationSolver.cpp"
        "${dir}/SolverHelper.h"
        "${dir}/SolverHelper.cpp"
        "${dir}/CuVISolver.h"
        "${dir}/CuVISolver.cpp"
        "${dir}/CudaOnlyValueIteration.cpp"
        "${dir}/CudaOnlyValueIteration.h"
        "${dir}/CuFunctions.cu"
        "${dir}/CuFunctions.h"
)

set(Sources "${Sources}" PARENT_SCOPE)