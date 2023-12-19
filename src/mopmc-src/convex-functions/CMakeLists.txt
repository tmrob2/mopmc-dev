set(dir "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(auxiliary)

list(
        APPEND Sources
        "${dir}/BaseConvexFunction.cpp"
        "${dir}/BaseConvexFunction.h"
        "${dir}/TotalReLU.cpp"
        "${dir}/TotalReLU.h"
        "${dir}/SignedKLEuclidean.cpp"
        "${dir}/SignedKLEuclidean.h"
        "${dir}/EuclideanDistance.cpp"
        "${dir}/EuclideanDistance.h"
)

set(Sources "${Sources}" PARENT_SCOPE)