#include "VulkanExtensions.h"
#include <stdexcept>

PFN_vkCmdSetFragmentShadingRateKHR pfnCmdSetFragmentShadingRateKHR = nullptr;

void loadVulkanExtensions(VkDevice device)
{
    pfnCmdSetFragmentShadingRateKHR =
        reinterpret_cast<PFN_vkCmdSetFragmentShadingRateKHR>(
            vkGetDeviceProcAddr(device, "vkCmdSetFragmentShadingRateKHR")
            );

    if (!pfnCmdSetFragmentShadingRateKHR) {
        throw std::runtime_error(
            "Failed to load vkCmdSetFragmentShadingRateKHR"
        );
    }
}
