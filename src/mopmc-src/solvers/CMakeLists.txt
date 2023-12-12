set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseValueIteration.h"
        "${dir}/CuFunctions.cu"
        "${dir}/CuFunctions.h"
        "${dir}/CudaValueIteration.cu"
        "${dir}/CudaValueIteration.cuh"
)

set(Sources "${Sources}" PARENT_SCOPE)