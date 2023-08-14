set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND TestSources
        "${dir}/matrix_tests.cpp"
)

set(TestSources "${TestSources}" PARENT_SCOPE)