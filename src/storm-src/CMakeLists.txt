set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ExplicitModelBuilder.cpp"
        "${dir}/ExplicitModelBuilder.h"
        "${dir}/TransitionMatrixBuilder.cpp"
        "${dir}/TransitionMatrixBuilder.h"
        "${dir}/RewardModelBuilder.h"
        "${dir}/RewardModelBuilder.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)