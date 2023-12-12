set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/AchievabilityQuery.cpp"
        "${dir}/AchievabilityQuery.h"
        "${dir}/BaseQuery.h"
        "${dir}/ConvexQuery.cpp"
        "${dir}/ConvexQuery.h"
        "${dir}/TestingQuery.cpp"
        "${dir}/TestingQuery.h"
)

set(Sources "${Sources}" PARENT_SCOPE)