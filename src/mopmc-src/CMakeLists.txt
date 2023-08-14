add_subdirectory(model-checking)
add_subdirectory(solvers)
set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ExplicitModelBuilder.cpp"
        "${dir}/ExplicitModelBuilder.h"
        #"${dir}/TransitionMatrixBuilder.cpp"
        #"${dir}/TransitionMatrixBuilder.h"
        #"${dir}/RewardModelBuilder.cpp"
        #"${dir}/RewardModelBuilder.h"
        "${dir}/SparseModel.cpp"
        "${dir}/SparseModel.h"
)

set(Sources "${Sources}" PARENT_SCOPE)
