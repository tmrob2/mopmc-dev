set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ExplicitModelBuilder.cpp"
        "${dir}/ExplicitModelBuilder.h"
        "${dir}/TransitionMatrixBuilder.cpp"
        "${dir}/TransitionMatrixBuilder.h"
        "${dir}/RewardModelBuilder.cpp"
        "${dir}/RewardModelBuilder.h"
)

set(Sources "${Sources}" PARENT_SCOPE)