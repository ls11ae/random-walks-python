cmake_minimum_required(VERSION 3.15)
project(random_walk LANGUAGES C)

# Add the C library project
add_subdirectory(random-walks)

# Copy the shared library to the Python package directory after build
install(TARGETS random_walk
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/random_walk_package
)


include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
