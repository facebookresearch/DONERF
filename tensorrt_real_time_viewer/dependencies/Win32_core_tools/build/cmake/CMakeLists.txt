cmake_minimum_required(VERSION 3.16)

project(Win32_core_tools)

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../source/")


file(GLOB Win32_core_tools_SOURCES "${SOURCE_DIR}/*.cpp")

include_directories(${GL_platform_tools_INCLUDE_DIRS} ${INCLUDE_DIR}Win32)

add_library(Win32_core_tools STATIC ${Win32_core_tools_SOURCES})

set(Win32_core_tools_INCLUDE_DIRS ${INCLUDE_DIR} CACHE INTERNAL "Win32 core tools include directories")
