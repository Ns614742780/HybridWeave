file(GLOB_RECURSE FWS ./ *.cpp)

# 添加 tinygltf 源文件
list(APPEND FWS ${CMAKE_SOURCE_DIR}/thirdParty/tinygltf/tiny_gltf.cc)

add_library(fws ${FWS})