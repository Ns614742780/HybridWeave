#include "Shader.h"
#include "Utils.h"
#include <stdexcept>
#include <vector>
#include <cstring>

void Shader::load()
{
    if (shader != VK_NULL_HANDLE) return;

    std::vector<char> shader_code;

    const uint32_t* codePtr = nullptr;
    size_t codeSize = 0;

    if (data == nullptr)
    {
        std::string filePath = "assets/shaders/" + filename + ".spv";
        shader_code = Utils::readFile(filePath);

        if (shader_code.empty()) {
            throw std::runtime_error("Failed to load shader: " + filePath);
        }

        codePtr = reinterpret_cast<const uint32_t*>(shader_code.data());
        codeSize = shader_code.size();
    }
    else
    {
        codePtr = reinterpret_cast<const uint32_t*>(data);
        codeSize = size;
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = codePtr;

    if (vkCreateShaderModule(context->device, &createInfo, nullptr, &shader) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module: " + filename);
    }

    // Debug name (optional)
    if (context->validationLayersEnabled && !filename.empty()) {
        VkDebugUtilsObjectNameInfoEXT nameInfo{};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
        nameInfo.objectHandle = reinterpret_cast<uint64_t>(shader);
        nameInfo.pObjectName = filename.c_str();

        auto func = (PFN_vkSetDebugUtilsObjectNameEXT)
            vkGetDeviceProcAddr(context->device, "vkSetDebugUtilsObjectNameEXT");

        if (func) {
            func(context->device, &nameInfo);
        }
    }
}

Shader::~Shader()
{
    if (shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context->device, shader, nullptr);
        shader = VK_NULL_HANDLE;
    }
}
