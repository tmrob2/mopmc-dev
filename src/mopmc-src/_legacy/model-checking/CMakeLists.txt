set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/SparseMultiObjective.h"
        "${dir}/SparseMultiObjective.cpp"
        "${dir}/MultiObjectivePreprocessor.cpp"
        "${dir}/MultiObjectivePreprocessor.h"
        "${dir}/MOPMCModelChecking.h"
        "${dir}/MOPMCModelChecking.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)