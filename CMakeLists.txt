cmake_minimum_required(VERSION 3.15)
project(random_walk LANGUAGES C)

include(GNUInstallDirs)

add_subdirectory(random-walks)

install(
    TARGETS random_walk
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
