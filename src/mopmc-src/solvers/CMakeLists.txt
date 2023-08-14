set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/LinearEquationUtils.cpp"
        "${dir}/LinearEquationUtils.h"
)

set(Sources "${Sources}" PARENT_SCOPE)