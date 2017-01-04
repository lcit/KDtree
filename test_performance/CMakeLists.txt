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

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories("/usr/local/include/eigen3/" ${OpenCV_INCLUDE_DIRS} ..)

# Declare the executable target built from your sources
add_executable(test_performance test_performance.cpp)

# Link your application with OpenCV libraries
target_link_libraries(test_performance ${OpenCV_LIBS})
