set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/BaseOptimizer.h"
        "${dir}/PolytopeTypeEnum.h"
        "${dir}/FrankWolfe.cpp"
        "${dir}/FrankWolfe.h"
        "${dir}/LinOpt.cpp"
        "${dir}/LinOpt.h"
        "${dir}/LineSearch.cpp"
        "${dir}/LineSearch.h"
        "${dir}/ProjectedGradientDescent.cpp"
        "${dir}/ProjectedGradientDescent.h"
)

set(Sources "${Sources}" PARENT_SCOPE)