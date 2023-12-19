add_subdirectory(model-checking)
add_subdirectory(solvers)
add_subdirectory(hybrid-computing)
add_subdirectory(queries)
add_subdirectory(convex-functions)
add_subdirectory(optimizers)
set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/Data.h"
        "${dir}/ExplicitModelBuilder.cpp"
        "${dir}/ExplicitModelBuilder.h"
        "${dir}/Runner.cpp"
        "${dir}/Runner.h"
        "${dir}/ModelBuilding.cpp"
        "${dir}/ModelBuilding.h"
        "${dir}/Transformation.cpp"
        "${dir}/Transformation.h"
        "${dir}/StormModelCheckingWrapper.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
