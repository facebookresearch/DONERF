cmake_minimum_required(VERSION 2.8)

project(test)

find_package(OpenGL REQUIRED)

if (WIN32)
	add_subdirectory(${Win32_core_tools_ROOT}/build/cmake Win32_core_tools)
	add_definitions(-DGLCORE_STATIC)
endif ()

add_subdirectory(../ GL_platform_tools)

include_directories(${GL_platform_tools_INCLUDE_DIRS})

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../test")

add_executable(test ${SOURCE_DIR}/main.cpp)

if (WIN32)
	target_link_libraries(test ${GL_platform_tools_LIBRARIES} Win32_core_tools)
else ()
	target_link_libraries(test ${GL_platform_tools_LIBRARIES})
endif ()
