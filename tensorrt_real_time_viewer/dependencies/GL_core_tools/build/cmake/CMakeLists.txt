cmake_minimum_required(VERSION 3.16)

project(GL_core_tools)

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../source/")


file(GLOB GL_core_tools_SOURCES "${SOURCE_DIR}/*.cpp")

include_directories(${GL_platform_tools_INCLUDE_DIRS} ${INCLUDE_DIR}GL)

if (WIN32)
include_directories(${Win32_core_tools_INCLUDE_DIRS})
endif ()

add_library(GL_core_tools STATIC ${GL_core_tools_SOURCES})

set(GL_core_tools_INCLUDE_DIRS ${INCLUDE_DIR} CACHE INTERNAL "GL core tools include directories")
