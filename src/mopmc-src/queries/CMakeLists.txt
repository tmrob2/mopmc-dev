set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/AchievabilityQuery.h"
        "${dir}/ConvQuery.h"
        "${dir}/ConvQuery.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)