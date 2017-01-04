cmake_minimum_required(VERSION 2.8)

project(KDtree_project)

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -O2")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

include_directories("/usr/local/include/eigen3/" ..)

add_executable(test_functional test_functional.cpp)

target_link_libraries(test_functional)
