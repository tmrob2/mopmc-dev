# Sub dir needs to be at the top of the parent file
add_subdirectory(storm-src)

set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
        APPEND Sources
        "${dir}/main.cpp"
)

set(Sources "${Sources}" PARENT_SCOPE)
