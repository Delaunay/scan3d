# required for C++11 features
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

PROJECT(scan3d CXX)

# debug/release
SET(CMAKE_RELEASE_POSTFIX "")                  # Release is the default
SET(CMAKE_DEBUG_POSTFIX "-debug")

# Compiler-flag
IF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    MESSAGE(STATUS "Building Debug Version")
ELSE()
    MESSAGE(STATUS "Building Release Version")
    # ADD_DEFINITIONS(-O2)                     # CMake Release is O3 by default
ENDIF()

# specify where compiled files must go
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# add cmake FindXXX script to find pre-compiled libraries if any
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# C++ Standard
# ====================================

SET(${CXX_STANDARD_REQUIRED} ON)    # Comment out if C++11 is not mendatory

# Project's Options
# ====================================

OPTION(BUILD_TESTING        "Enable tests"       OFF)
OPTION(BUILD_BENCHMARK      "Build Benchmarks"   OFF)
OPTION(BUILD_EXAMPLES       "Build Examples"     OFF)
OPTION(BUILD_DOCUMENTATION  "Build docs"         OFF)
OPTION(BUILD_DOXYGEN        "Build Doxygen docs" OFF)
OPTION(BUILD_SPHINX         "Build Sphinx docs"  OFF)

# Binary/pre-compiled Dependencies
# ====================================

# SET(OPENCV_DIR C:/Users/Setepenre/dist/opencv)
FIND_PACKAGE(OpenCV REQUIRED core photo video imgproc calib3d highgui)

# Subdirectories
# ====================================

# Find revision ID and hash of the sourcetree
# FIND_PACKAGE(Git REQUIRED)
# INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})
# INCLUDE(cmake/genrev.cmake)

# Dependencies to be compiled
# ADD_SUBDIRECTORY(dependencies)

# How to compile main lib
ADD_SUBDIRECTORY(src)

IF(BUILD_TESTING)
    ENABLE_TESTING()
	
    MESSAGE(STATUS "Building tests")
    ADD_SUBDIRECTORY(tests)
ENDIF(BUILD_TESTING)

IF (BUILD_EXAMPLES)
    MESSAGE(STATUS "Building examples")
    ADD_SUBDIRECTORY(examples)
ENDIF (BUILD_EXAMPLES)

IF (BUILD_DOCUMENTATION)
    MESSAGE(STATUS "Documentation building is enabled")
    ADD_SUBDIRECTORY(docs)
ENDIF (BUILD_DOCUMENTATION)



