cmake_minimum_required(VERSION 3.2)

# set the project name and version
project(mopmc LANGUAGES CXX C CUDA VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# search for Storm library
#find_package(storm REQUIRED)
#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#set(storm_INCLUDE_DIR, "/home/thomas/storm/build/storm")
#set(storm-parsers_INCLUDE_DIR, "/home/thomas/storm/build/storm-parsers")
# specify source files
set(Sources)
set(TestSources)
add_subdirectory(src)
add_subdirectory(lpSolve)
#add_subdirectory(test)
# set executable
add_executable(${PROJECT_NAME} "${Sources}")
find_package(storm REQUIRED)
find_package(CUDAToolkit REQUIRED)
#set(ProjectTests "${PROJECT_NAME}_tests")
#set(EigenSource /home/thomas/eigen-3.4.0)
#add_executable(ProjectTests "${TestSources}")

enable_testing()

# Google test framework
add_subdirectory(googletest)

set(LibHeaders "src/mopmc-src/solvers/ActionSelection.h")
set(LibSources "src/mopmc-src/solvers/ActionSelection.cu")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
add_library(libmopmc STATIC ${LibSources} ${LibHeaders})
target_link_libraries(libmopmc PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)
set_target_properties(libmopmc PROPERTIES LINKER_LANGUAGE CUDA CUDA_ARCHITECTURES OFF)

#set_target_properties(libmopmc PROPERTIES CUDA_ARCHITECTURES 70-real 72-virtual)

# Set include directories and dependencies
target_include_directories(${PROJECT_NAME} PUBLIC ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} lpSolve PRIVATE src)
target_link_directories(${PROJECT_NAME} PUBLIC lpSolve)
target_link_libraries(${PROJECT_NAME} PRIVATE libmopmc storm storm-parsers lpsolve55 CUDA::cudart CUDA::cusparse CUDA::cublas)
#target_include_directories(ProjectTests PUBLIC ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} EigenSources PRIVATE test)
#target_link_libraries(ProjectTests GTest::gtest_main storm storm-parsers)

#include(GoogleTest)
#gtest_discover_tests(ProjectTests)
# Note that other libraries/includes from Storm might need to be added depending on the used functionality.
# For example, to enable counterexamples, include directory ${storm-counterexamples_INCLUDE_DIR} and link library storm-counterexamples need to be added.
