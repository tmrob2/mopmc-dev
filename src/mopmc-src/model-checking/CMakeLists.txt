set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/ModelChecker.cpp"
        "${dir}/ModelChecker.h"
        "${dir}/ModelCheckingUtils.cpp"
        "${dir}/ModelCheckingUtils.h"
)

set(Sources "${Sources}" PARENT_SCOPE)