#pragma once

#include <memory>
#include <string>
#include <utility>

#include "VulkanContext.h"

class Shader {
public:
    Shader(const std::shared_ptr<VulkanContext>& _context, std::string filename)
        : context(_context),
        filename(std::move(filename)) {
    }

    Shader(const std::shared_ptr<VulkanContext>& _context,
        const unsigned char* data, size_t size)
        : context(_context), filename(""), data(data), size(size) {
    }

    Shader(const std::shared_ptr<VulkanContext>& _context,
        const std::string& filename,
        const unsigned char* data, size_t size)
        : context(_context), filename(filename), data(data), size(size) {
    }

    void load();

    // Vulkan C API shader module
    VkShaderModule shader = VK_NULL_HANDLE;

    ~Shader();

private:
    const std::string filename;
    std::shared_ptr<VulkanContext> context;
    const unsigned char* data = nullptr;
    size_t size = 0;
};
