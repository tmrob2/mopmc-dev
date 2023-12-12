set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ActionSelection.cu"
        "${dir}/ActionSelection.h"
        "${dir}/ConvexQuery.cpp"
        "${dir}/ConvexQuery.h"
        "${dir}/CudaOnlyValueIteration.cpp"
        "${dir}/CudaOnlyValueIteration.h"
        "${dir}/CuVISolver.cpp"
        "${dir}/CuVISolver.h"
        "${dir}/InducedEquationSolver.cpp"
        "${dir}/InducedEquationSolver.h"
        "${dir}/IterativeSolver.cpp"
        "${dir}/IterativeSolver.h"
        "${dir}/SolverHelper.cpp"
        "${dir}/SolverHelper.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
