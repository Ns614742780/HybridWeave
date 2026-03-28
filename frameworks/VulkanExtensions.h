#pragma once
#include <vulkan/vulkan.h>

// KHR_fragment_shading_rate
extern PFN_vkCmdSetFragmentShadingRateKHR pfnCmdSetFragmentShadingRateKHR;

void loadVulkanExtensions(VkDevice device);