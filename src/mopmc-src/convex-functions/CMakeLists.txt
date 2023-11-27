set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseConvexFunction.h"
        "${dir}/TotalReLU.cpp"
        "${dir}/TotalReLU.h"
        "${dir}/SignedKLEuclidean.cpp"
        "${dir}/SignedKLEuclidean.h"
)

set(Sources "${Sources}" PARENT_SCOPE)