set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/AchievabilityQuery.cpp"
        "${dir}/AchievabilityQuery.h"
        "${dir}/BaseQuery.h"
        "${dir}/GpuConvexQuery.cpp"
        "${dir}/GpuConvexQuery.h"
        "${dir}/GpuConvexQueryAlt.cpp"
        "${dir}/GpuConvexQueryAlt.h"
        "${dir}/TestingQuery.cpp"
        "${dir}/TestingQuery.h"
        "${dir}/HybridQuery.h"
        "${dir}/HybridQuery.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)