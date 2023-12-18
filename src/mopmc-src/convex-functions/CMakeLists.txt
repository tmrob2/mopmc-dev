set(dir "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(auxiliary)

list(
        APPEND Sources
        "${dir}/BaseConvexFunction.cpp"
        "${dir}/BaseConvexFunction.h"
        "${dir}/EuclideanDistance.cpp"
        "${dir}/EuclideanDistance.h"
        "${dir}/KLDistance.cpp"
        "${dir}/KLDistance.h"
        "${dir}/MSE.cpp"
        "${dir}/MSE.h"
        "${dir}/SignedKLEuclidean.cpp"
        "${dir}/SignedKLEuclidean.h"
        "${dir}/TotalReLU.cpp"
        "${dir}/TotalReLU.h"
)

set(Sources "${Sources}" PARENT_SCOPE)