#pragma once
#include <vulkan/vulkan.h>
#include <memory>

struct Image;

// provide a color image for compositing
class IColorProvider {
public:
    virtual ~IColorProvider() = default;

    // per swapchain image index
    virtual const std::shared_ptr<Image>& getColorImage(uint32_t imageIndex) const = 0;
    virtual VkImageLayout getColorLayout(uint32_t imageIndex) const = 0;

    // composite
    virtual VkFormat getColorFormat() const = 0;
    virtual VkExtent2D getColorExtent() const = 0;
};
