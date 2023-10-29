set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/HybridThreadLib.cpp"
        "${dir}/HybridThreadLib.h"
)

set(Sources "${Sources}" PARENT_SCOPE)