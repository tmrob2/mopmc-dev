add_subdirectory(solvers)
add_subdirectory(hybrid-computing)
add_subdirectory(queries)
add_subdirectory(convex-functions)
add_subdirectory(optimizers)
add_subdirectory(storm-wrappers)
add_subdirectory(_legacy)
set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/Data.h"
        "${dir}/Runner.cpp"
        "${dir}/Runner.h"
        "${dir}/Transformation.cpp"
        "${dir}/Transformation.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
