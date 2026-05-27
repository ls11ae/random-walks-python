cmake_minimum_required(VERSION 3.18)
project(random_walk_python LANGUAGES C CXX)

include(GNUInstallDirs)

add_subdirectory(random-walks)

install(TARGETS random_walk
    COMPONENT random_walk
    LIBRARY DESTINATION random_walk_package
    RUNTIME DESTINATION random_walk_package
    ARCHIVE DESTINATION random_walk_package
)