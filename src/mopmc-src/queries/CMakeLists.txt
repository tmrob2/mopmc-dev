set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/AchievabilityQuery.h"
        "${dir}/ConvexxQuery.h"
)

set(Sources "${Sources}" PARENT_SCOPE)