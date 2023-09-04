set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/OptimalPolicy.cpp"
        "${dir}/OptimalPolicy.h"
)

set(Sources "${Sources}" PARENT_SCOPE)