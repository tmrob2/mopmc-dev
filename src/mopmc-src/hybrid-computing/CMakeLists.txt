set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/Looper.cpp"
        "${dir}/Looper.h"
        "${dir}/Problem.h"
        "${dir}/Problem.cpp"
        "${dir}/HybridController.h"
        "${dir}/HybridController.cpp"
        "${dir}/Utilities.h"
        "${dir}/Utilities.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)