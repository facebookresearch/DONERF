cmake_minimum_required(VERSION 3.16)

project(GL_platform_tools)

find_package(OpenGL REQUIRED)

set(INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../source")

if (WIN32)
	set(PLATFORM_SOURCE_DIR "${SOURCE_DIR}/win32")
	set(INCLUDE_DIRS "${PLATFORM_SOURCE_DIR}/include" ${INCLUDE_DIRS})
	set(INCLUDE_DIRS ${Win32_core_tools_INCLUDE_DIRS} ${INCLUDE_DIRS})
else ()
	find_package(X11 REQUIRED)
	set(PLATFORM_SOURCE_DIR "${SOURCE_DIR}/x11")
	set(INCLUDE_DIRS "${PLATFORM_SOURCE_DIR}/include" ${INCLUDE_DIRS})
	set(INCLUDE_DIRS "${SOURCE_DIR}" ${INCLUDE_DIRS})
endif ()

include_directories(${INCLUDE_DIRS})

if (WIN32)
	add_definitions(-DGLCORE_STATIC)
	add_subdirectory(glcore)
endif ()

file(GLOB GL_platform_tools_SOURCES "${SOURCE_DIR}/*.cpp" "${PLATFORM_SOURCE_DIR}/*.cpp")

add_library(GL_platform_tools STATIC ${GL_platform_tools_SOURCES})

set(GL_platform_tools_INCLUDE_DIRS ${INCLUDE_DIRS} CACHE INTERNAL "GL platform tools include directories")

if (WIN32)
	set(GL_platform_tools_LIBRARIES GL_platform_tools glcore ${OPENGL_gl_LIBRARY} CACHE INTERNAL "GL platform tools include directories")
else ()
	set(GL_platform_tools_LIBRARIES GL_platform_tools ${OPENGL_gl_LIBRARY} ${X11_LIBRARIES} CACHE INTERNAL "GL platform tools include directories")
endif ()
