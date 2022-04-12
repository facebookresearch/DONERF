cmake_minimum_required(VERSION 3.16)
project(nr-real-time LANGUAGES CXX CUDA VERSION 1.0)

#####################################################################################
# Options
set(CC "86" CACHE STRING "Which compute capability to build for? E.g. 61, 70, 75")

#####################################################################################
# Packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
if(WIN32)
else()
find_package(X11 REQUIRED)
endif()

#####################################################################################
# Dependencies
if (WIN32)
  add_subdirectory(dependencies/Win32_core_tools/build/cmake Win32_core_tools)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX)
endif()
  add_definitions(-DGLCORE_STATIC)
  add_subdirectory(dependencies/GL_platform_tools/build/cmake GL_platform_tools)
  add_subdirectory(dependencies/GL_core_tools/build/cmake GL_core_tools)
  

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_NVONNX_PARSER nvonnxparser
   HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
   PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_NVONNX_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARIES}")
find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARIES)

set(DEPENDENCIES_DIR "${CMAKE_SOURCE_DIR}/dependencies")
get_filename_component(TENSORRT_ROOT ${TENSORRT_INCLUDE_DIR}/.. ABSOLUTE)
set(TENSORRT_SAMPLES_COMMON_DIR "${TENSORRT_ROOT}/samples/common")
MESSAGE(STATUS "TensorRT samples common at ${TENSORRT_SAMPLES_COMMON_DIR}")

set(DEPENDENCIES_INCLUDE_DIRS 
  "${DEPENDENCIES_DIR}/GL_platform_tools/include"
  "${DEPENDENCIES_DIR}/Win32_core_tools/include"
  "${DEPENDENCIES_DIR}/GL_core_tools/include"
  "${DEPENDENCIES_DIR}/GL_core_tools/include/GL"
  "${DEPENDENCIES_DIR}/glm"
  "${DEPENDENCIES_DIR}/argparse"
  "${GL_platform_tools_INCLUDE_DIRS}"
  "${TENSORRT_SAMPLES_COMMON_DIR}"
)
include("${CMAKE_SOURCE_DIR}/dependencies/GLSL_build_tools/build/cmake/add_glsl_sources.cmake")

####################################################################
# GPU Implementation
add_library(donerf_gpu
   include/cuda/donerf_cuda_kernels.cuh

   src/cuda/base_cuda_kernels.cu
   src/cuda/coarse_cuda_kernels.cu
)

target_include_directories(donerf_gpu PRIVATE 
  "${CMAKE_SOURCE_DIR}/include" 
  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" 
  "${DEPENDENCIES_INCLUDE_DIRS}")

set_target_properties(donerf_gpu PROPERTIES CUDA_ARCHITECTURES ${CC})
set_target_properties(donerf_gpu PROPERTIES CXX_STANDARD 20)
target_compile_options(donerf_gpu PRIVATE -Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v -lineinfo $<$<CONFIG:DEBUG>:-G>)


#####################################################################################
# Executable
add_executable(donerf

  # Source Files
  src/basicbuffermanager.cpp
  src/camera.cpp
  src/config.cpp
  src/encoding.cpp
  src/featureset.cpp
  src/imagegenerator.cpp
  src/inputhandler.cpp
  src/interoprenderbuffer.cpp
  src/main.cpp
  src/neuralrenderer.cpp
  src/settings.cpp
  ${TENSORRT_SAMPLES_COMMON_DIR}/logger.cpp

  # Include Files
  include/basicbuffermanager.h
  include/bmp.h
  include/buffermanager.h
  include/camera.h
  include/config.h
  include/encoding.h
  include/featureset.h
  include/helper.h
  include/imagegenerator.h
  include/inputhandler.h
  include/interoprenderbuffer.h
  include/neuralrenderer.h
  include/settings.h
)

if(NOT TENSORRT_FOUND)
  message(ERROR "Cannot find TensorRT library.")
endif()

target_include_directories(donerf PRIVATE 
  "${DEPENDENCIES_INCLUDE_DIRS}" 
  "${CMAKE_SOURCE_DIR}/include" 
  "${CMAKE_SOURCE_DIR}/dependencies"
  "${TENSORRT_INCLUDE_DIR}"
  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")


set_target_properties(donerf PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(donerf PROPERTIES CXX_STANDARD 17)
if (MSVC)
   target_compile_options(donerf PRIVATE /W4 $<$<CONFIG:RELEASE>:/O2> $<$<CONFIG:DEBUG>:/Od>)
else()
   target_compile_options(donerf PRIVATE -Wall -Wextra $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:DEBUG>:-O0>)
endif()

if (WIN32)
	target_link_libraries(donerf donerf_gpu Win32_core_tools ${GL_platform_tools_LIBRARIES} ${TENSORRT_LIBRARIES} ${instrumentation_LIBRARIES} GL_core_tools ${OPENGL_LIBRARIES})
else ()
	target_link_libraries(donerf donerf_gpu ${X11_LIBRARIES} ${TENSORRT_LIBRARIES} ${CMAKE_DL_LIBS} ${GL_platform_tools_LIBRARIES} GL_core_tools "-lX11" ${OPENGL_LIBRARIES})
endif ()