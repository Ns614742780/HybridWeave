cmake_minimum_required(VERSION 3.12)
project(HybridWeave)
set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    add_compile_options("/utf-8")
endif()

set(VULKAN_ROOT ${CMAKE_SOURCE_DIR}/thirdParty/vulkan)

set(VULKAN_INCLUDE_DIR ${VULKAN_ROOT}/include)
set(VULKAN_LIBRARY ${VULKAN_ROOT}/lib/vulkan-1.lib)
set(GLSLC_EXE ${VULKAN_ROOT}/bin/glslc.exe)

message(STATUS "Using Vulkan Include: ${VULKAN_INCLUDE_DIR}")
message(STATUS "Using Vulkan Library: ${VULKAN_LIBRARY}")
message(STATUS "Using glslc compiler : ${GLSLC_EXE}")

include_directories(
    SYSTEM ${CMAKE_SOURCE_DIR}/thirdParty/include
    ${VULKAN_INCLUDE_DIR}
)

link_directories(
    ${CMAKE_SOURCE_DIR}/thirdParty/lib
    ${VULKAN_ROOT}/lib
)

file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})

add_subdirectory(frameworks)

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/assets/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/assets/shaders)

file(GLOB_RECURSE SHADERS
    "${SHADER_SRC_DIR}/*.vert"
    "${SHADER_SRC_DIR}/*.frag"
    "${SHADER_SRC_DIR}/*.comp"
)

set(SPV_FILES)

foreach(shader ${SHADERS})
    file(RELATIVE_PATH rel_path "${SHADER_SRC_DIR}" "${shader}")

    string(REGEX REPLACE "\\.[^.]*$" "" rel_no_ext "${rel_path}")

    set(output_file "${SHADER_OUT_DIR}/${rel_no_ext}.spv")

    get_filename_component(output_dir "${output_file}" DIRECTORY)

    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
        COMMAND ${GLSLC_EXE} --target-env=vulkan1.3 ${shader} -o ${output_file}
        DEPENDS ${shader}
        COMMENT "Compiling shader: ${rel_path}"
    )

    list(APPEND SPV_FILES ${output_file})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SPV_FILES})

add_executable(HybridWeave "main.cpp")

target_link_libraries(HybridWeave
    fws
    glfw3.lib
    ${VULKAN_LIBRARY}
)

add_dependencies(HybridWeave Shaders)
