set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        #"${dir}/ModelChecker.cpp"
        #"${dir}/ModelChecker.h"
        #"${dir}/GraphAnalysis.cpp"
        #"${dir}/GraphAnalysis.h"
        #"${dir}/MultiObjective.cpp"
        #"${dir}/MultiObjective.h"
        "${dir}/MultiObjectiveTest.h"
        "${dir}/MultiObjectiveTest.cpp"
        "${dir}/MultiObjectivePreprocessor.cpp"
        "${dir}/MultiObjectivePreprocessor.h"
        "${dir}/StandardMdpPcaaChecker.h"
        "${dir}/StandardMdpPcaaChecker.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)