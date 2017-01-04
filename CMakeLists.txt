# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(KDtree_project)

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -O2")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

#FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

include_directories("/usr/local/include/eigen3/" ..)

# Declare the executable target built from your sources
add_executable(main main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(main)
