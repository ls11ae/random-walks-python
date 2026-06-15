cmake_minimum_required(VERSION 3.18)
project(random_walk_python LANGUAGES C CXX)

include(GNUInstallDirs)

add_subdirectory(random-walks)

install(TARGETS random_walk
    COMPONENT random_walk
    LIBRARY DESTINATION randomwalks
    RUNTIME DESTINATION randomwalks
    ARCHIVE DESTINATION randomwalks
)