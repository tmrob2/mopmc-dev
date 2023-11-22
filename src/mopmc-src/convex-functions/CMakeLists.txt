set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/TotalReLU.cpp"
        "${dir}/TotalReLU.h"
)

set(Sources "${Sources}" PARENT_SCOPE)