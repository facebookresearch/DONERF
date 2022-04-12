cmake_minimum_required(VERSION 3.16)


add_executable(glsl2cpp ${CMAKE_CURRENT_LIST_DIR}/../../source/glsl2cpp.cpp)


macro(glsl_add_definitions sources defs)
  set_property(SOURCE ${sources} APPEND PROPERTY GLSL_DEFINITIONS ${defs})
endmacro()

macro(glsl_add_include_directories sources dirs)
  set_property(SOURCE ${sources} APPEND PROPERTY GLSL_INCLUDE_DIRS ${dirs})
endmacro()

function(glsl_source target source)
  get_property(idirs SOURCE ${source} PROPERTY GLSL_INCLUDE_DIRS)
  foreach (i ${idirs})
    list(APPEND cmdline -I ${i})
  endforeach()
  
  get_property(defs SOURCE ${source} PROPERTY GLSL_DEFINITIONS)
  foreach (d ${defs})
    list(APPEND cmdline -D${d})
  endforeach()
  
  add_custom_command(
    OUTPUT  ${target}
    COMMAND glsl2cpp
    ARGS    ${cmdline} -o ${target} ${source}
    MAIN_DEPENDENCY ${source}
  )
endfunction()

function(add_glsl_sources library sources)
  foreach (f ${sources})
    get_filename_component(n ${f} NAME)
    glsl_source(${n}.cpp ${f})
    list(APPEND shader_files ${n}.cpp)
  endforeach()  
  add_library(${library} ${shader_files})
endfunction()
