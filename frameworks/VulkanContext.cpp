#include "VulkanContext.h"
#include "VulkanExtensions.h"
#include "Utils.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <set>
#include <cstring>

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData)
{
    const char* t = (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) ? "VALIDATION" :
        (type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) ? "PERFORMANCE" :
        (type & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) ? "GENERAL" : "UNKNOWN";

    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        spdlog::error("[{}] {}", t, callbackData->pMessage);
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        spdlog::warn("[{}] {}", t, callbackData->pMessage);
    else
        spdlog::info("[{}] {}", t, callbackData->pMessage);

    return VK_FALSE;
}

VulkanContext::VulkanContext(const std::vector<std::string>& instance_extensions,
    const std::vector<std::string>& device_extensions,
    bool validationEnabled)
    : instanceExtensions(instance_extensions),
    deviceExtensions(device_extensions),
    validationLayersEnabled(validationEnabled)
{
    // Add dynamic rendering extension
    deviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

    if (validationLayersEnabled) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
}

void VulkanContext::createInstance()
{
    // Application info
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Splatting";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // Validation layer
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* layers[] = { layer };

    std::vector<const char*> extensions =
        Utils::stringVectorToCharPtrVector(instanceExtensions);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = validationLayersEnabled ? 1 : 0;
    createInfo.ppEnabledLayerNames = validationLayersEnabled ? layers : nullptr;
    createInfo.enabledExtensionCount = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Debug messenger
    VkDebugUtilsMessengerCreateInfoEXT debugCreate{};
    if (validationLayersEnabled) {
        debugCreate = {};
        debugCreate.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreate.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;

        debugCreate.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

        debugCreate.pfnUserCallback = debugCallback;

        createInfo.pNext = &debugCreate;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan instance");

    spdlog::debug("Vulkan instance created");
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice device, std::optional<VkSurfaceKHR> surf)
{
    // Check required extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    for (auto& ext : deviceExtensions) {
        bool found = false;
        for (auto& a : available) {
            if (strcmp(ext.c_str(), a.extensionName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }

    if (surf.has_value()) {
        uint32_t formatCount = 0, modeCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surf.value(), &formatCount, nullptr);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surf.value(), &modeCount, nullptr);
        if (formatCount == 0 || modeCount == 0) return false;
    }

    return true;
}

void VulkanContext::selectPhysicalDevice(std::optional<uint8_t> id,
    std::optional<VkSurfaceKHR> surf)
{
    if (surf.has_value())
        surface = surf;

    // Enumerate physical devices
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0)
        throw std::runtime_error("No Vulkan physical device found");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    spdlog::info("Available devices:");
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[i], &props);
        spdlog::info("[{}] {}", i, props.deviceName);
    }

    if (surf.has_value())
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    if (id.has_value()) {
        if (id.value() >= devices.size())
            throw std::runtime_error("Invalid device id");

        physicalDevice = devices[id.value()];
        return;
    }

    // Auto choose discrete GPU if possible
    for (auto d : devices) {
        if (isDeviceSuitable(d, surf)) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice = d;
                return;
            }
        }
    }

    // Fallback: first suitable
    for (auto d : devices) {
        if (isDeviceSuitable(d, surf)) {
            physicalDevice = d;
            return;
        }
    }

    throw std::runtime_error("No suitable GPU found");
}

VulkanContext::QueueFamilyIndices VulkanContext::findQueueFamilies()
{
    QueueFamilyIndices indices;
    uint32_t count = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());

    for (uint32_t i = 0; i < count; i++) {
        const auto& q = families[i];

        if (q.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;

        if (q.queueFlags & VK_QUEUE_COMPUTE_BIT)
            indices.computeFamily = i;

        if (surface.has_value()) {
            VkBool32 presentSupported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface.value(), &presentSupported);
            if (presentSupported)
                indices.presentFamily = i;
        }

        if (indices.isComplete())
            break;
    }

    return indices;
}

void VulkanContext::createLogicalDevice(
    VkPhysicalDeviceFeatures features,
    VkPhysicalDeviceVulkan11Features features11,
    VkPhysicalDeviceVulkan12Features features12)
{
    auto families = findQueueFamilies();
    std::set<uint32_t> unique = {
        families.graphicsFamily.value(),
        families.computeFamily.value(),
        families.presentFamily.value()
    };

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    for (uint32_t fam : unique) {
        VkDeviceQueueCreateInfo q{};
        q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q.queueFamilyIndex = fam;
        q.queueCount = 1;
        q.pQueuePriorities = &priority;
        queueInfos.push_back(q);
    }

    std::vector<const char*> devExt =
        Utils::stringVectorToCharPtrVector(deviceExtensions);

    features.samplerAnisotropy = VK_TRUE;
    features.independentBlend = VK_TRUE;

    auto hasDeviceExt = [&](const char* name) {
        return std::any_of(
            devExt.begin(), devExt.end(),
            [&](const char* s) { return strcmp(s, name) == 0; }
        );
    };

    bool enableVrs = false;
    {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, exts.data());

        for (auto& e : exts) {
            if (strcmp(e.extensionName, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME) == 0) {
                enableVrs = true;
                if (!hasDeviceExt(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) {
                    devExt.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
                }
                break;
            }
        }
    }

    // pNext chain
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.descriptorIndexing = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;

    VkDeviceCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create.queueCreateInfoCount = queueInfos.size();
    create.pQueueCreateInfos = queueInfos.data();
    create.enabledExtensionCount = devExt.size();
    create.ppEnabledExtensionNames = devExt.data();
    create.pEnabledFeatures = &features;
    create.pNext = &features11;
    features11.pNext = &features12;

    VkPhysicalDeviceDynamicRenderingFeaturesKHR dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dyn.dynamicRendering = VK_TRUE;

    features12.pNext = &dyn;

    VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrs{};
    if (enableVrs) {
        vrs.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
        vrs.pipelineFragmentShadingRate = VK_TRUE;
        vrs.primitiveFragmentShadingRate = VK_FALSE;
        vrs.attachmentFragmentShadingRate = VK_TRUE;

        dyn.pNext = &vrs;
    }

    if (vkCreateDevice(physicalDevice, &create, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");

    // ---- query fragment shading rate properties ----
    VkPhysicalDeviceFragmentShadingRatePropertiesKHR vrsProps{};
    vrsProps.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &vrsProps;

    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    VkExtent2D rateTexelSize = vrsProps.maxFragmentShadingRateAttachmentTexelSize;

    fragmentShadingRateTexelSize = rateTexelSize;

    VkFormatProperties3 props3{
    VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3
    };
    VkFormatProperties2 propsFmt2{
        VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2
    };
    propsFmt2.pNext = &props3;

    vkGetPhysicalDeviceFormatProperties2(
        physicalDevice,
        VK_FORMAT_R8_UINT,
        &propsFmt2
    );

    if ((props3.optimalTilingFeatures &
        VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR) == 0) {
        throw std::runtime_error(
            "VK_FORMAT_R8_UINT does not support fragment shading rate attachment"
        );
    }

    loadVulkanExtensions(device);

    // Get device queues
    for (uint32_t fam : unique) {
        VkQueue q;
        vkGetDeviceQueue(device, fam, 0, &q);

        std::set<Queue::Type> types;
        if (fam == families.graphicsFamily.value()) types.insert(Queue::GRAPHICS);
        if (fam == families.computeFamily.value())  types.insert(Queue::COMPUTE);
        if (fam == families.presentFamily.value())  types.insert(Queue::PRESENT);

        for (auto t : types) {
            queues[t] = Queue{ types, fam, 0, q };
        }
    }

    setupVma();
    createCommandPool();
    createQueryPool();
}

VkCommandBuffer VulkanContext::beginOneTimeCommandBuffer()
{
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

void VulkanContext::endOneTimeCommandBuffer(VkCommandBuffer cmd, Queue::Type type)
{
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    vkQueueSubmit(queues[type].queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queues[type].queue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

void VulkanContext::setupVma()
{
    VmaAllocatorCreateInfo info{};
    info.physicalDevice = physicalDevice;
    info.device = device;
    info.instance = instance;

    vmaCreateAllocator(&info, &allocator);
}

void VulkanContext::createCommandPool()
{
    auto families = findQueueFamilies();

    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool.queueFamilyIndex = families.graphicsFamily.value();

    vkCreateCommandPool(device, &pool, nullptr, &commandPool);
}

void VulkanContext::createQueryPool()
{
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = 20;

    vkCreateQueryPool(device, &info, nullptr, &queryPool);

    // Reset queries
    VkCommandBuffer cmd = beginOneTimeCommandBuffer();
    vkCmdResetQueryPool(cmd, queryPool, 0, 12);
    endOneTimeCommandBuffer(cmd, Queue::GRAPHICS);
}

void VulkanContext::createDescriptorPool(uint8_t frames)
{
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames * 32 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frames * 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  frames * 256 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frames * 512 },
    };

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = 4;
    info.pPoolSizes = sizes;
    info.maxSets = 2048;
    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool);
}

VulkanContext::~VulkanContext()
{
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }

    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }

    if (queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, queryPool, nullptr);
        queryPool = VK_NULL_HANDLE;
    }

    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    if (allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }

    if (surface.has_value()) {
		VkSurfaceKHR tmp = surface.value();
        vkDestroySurfaceKHR(instance, tmp, nullptr);
        surface = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

