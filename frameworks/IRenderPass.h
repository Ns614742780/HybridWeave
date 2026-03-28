#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

class Swapchain;

class IRenderPass
{
public:
    virtual ~IRenderPass() = default;

	// create & initialize resources
    virtual void initialize() = 0;

	// called when swapchain is resized
    virtual void onSwapchainResized() = 0;

	// update per frame (e.g., update uniforms)
    virtual void update(float dt) {}

	// record command buffer for this pass
    virtual void record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex) = 0;
};
