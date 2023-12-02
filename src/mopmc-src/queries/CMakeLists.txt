set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/AchievabilityQuery.cpp"
        "${dir}/AchievabilityQuery.h"
        "${dir}/BaseQuery.h"
        "${dir}/GpuConvexQuery.cpp"
        "${dir}/GpuConvexQuery.h"
        "${dir}/TestingQuery.cpp"
        "${dir}/TestingQuery.h"
)

set(Sources "${Sources}" PARENT_SCOPE)