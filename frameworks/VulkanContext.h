#pragma once

#define FRAMES_IN_FLIGHT 1

#include <optional>
#include <set>
#include <unordered_map>
#include <vector>
#include <string>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#include "Image.h"

class VulkanContext {
private:
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> computeFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() const {
            return graphicsFamily.has_value() &&
                computeFamily.has_value() &&
                presentFamily.has_value();
        }
    };

public:
    struct Queue {
        enum Type {
            GRAPHICS,
            COMPUTE,
            PRESENT
        };

        std::set<Type> types;
        uint32_t queueFamily;
        uint32_t queueIndex;
        VkQueue queue;
    };

    VulkanContext(const std::vector<std::string>& instanceExtensions,
        const std::vector<std::string>& deviceExtensions,
        bool validationLayersEnabled);

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

    // --- API consistent with original version ---
    void createInstance();
    bool isDeviceSuitable(VkPhysicalDevice device, std::optional<VkSurfaceKHR> surface = std::nullopt);
    void selectPhysicalDevice(std::optional<uint8_t> id = std::nullopt,
        std::optional<VkSurfaceKHR> surface = std::nullopt);

    QueueFamilyIndices findQueueFamilies();
    void createQueryPool();
    void createLogicalDevice(VkPhysicalDeviceFeatures deviceFeatures,
        VkPhysicalDeviceVulkan11Features deviceFeatures11,
        VkPhysicalDeviceVulkan12Features deviceFeatures12);
    void createDescriptorPool(uint8_t framesInFlight);

    VkCommandBuffer beginOneTimeCommandBuffer();
    void endOneTimeCommandBuffer(VkCommandBuffer commandBuffer, Queue::Type queue);

    virtual ~VulkanContext();

    // --- Members (now plain Vulkan handles instead of vk::) ---
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    std::optional<VkSurfaceKHR> surface;

    VkDevice device = VK_NULL_HANDLE;
    std::unordered_map<Queue::Type, Queue> queues;
    VmaAllocator allocator = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;

    VkExtent2D fragmentShadingRateTexelSize{ 1, 1 };

    bool validationLayersEnabled = false;

private:
    std::vector<std::string> instanceExtensions;
    std::vector<std::string> deviceExtensions;

    VkCommandPool commandPool = VK_NULL_HANDLE;

    // C API version
    void setupVma();
    void createCommandPool();
};

