cmake_minimum_required(VERSION 2.8)

project(glcore)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../source/win32/glcore")

file(GLOB glcore_SOURCES "${SOURCE_DIR}/*.cpp")

add_library(glcore STATIC ${glcore_SOURCES})

set(glcore_INCLUDE_DIRS ${INCLUDE_DIRS} CACHE INTERNAL "glcore include directories")
