add_subdirectory(model-checking)
add_subdirectory(solvers)
add_subdirectory(hybrid-computing)
add_subdirectory(queries)

set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ExplicitModelBuilder.cpp"
        "${dir}/ExplicitModelBuilder.h"
        "${dir}/Runner.cpp"
        "${dir}/Runner.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
