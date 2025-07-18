cmake_minimum_required(VERSION 3.15)
project(random_walk LANGUAGES C CXX)

# Standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Include directories
include_directories(
        ${R_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src
)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Gather sources
file(GLOB_RECURSE SOURCES
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

file(GLOB_RECURSE HEADERS
        "${PROJECT_SOURCE_DIR}/src/*.h"
)

# Main library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        OpenMP::OpenMP_C
        OpenMP::OpenMP_CXX
        ${R_LIBRARIES}
)

# Executable (if needed)
add_executable(${PROJECT_NAME}_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main PRIVATE ${PROJECT_NAME})

# Tests
enable_testing()

install(TARGETS random_walk
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../random_walk_package
)

