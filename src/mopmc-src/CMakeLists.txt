add_subdirectory(model-checking)
add_subdirectory(solvers)
add_subdirectory(hybrid-computing)
set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ExplicitModelBuilder.cpp"
        "${dir}/ExplicitModelBuilder.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
