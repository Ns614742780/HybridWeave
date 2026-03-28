#include "Swapchain.h"

#include <algorithm>
#include <vulkan/vk_enum_string_helper.h>
#include <spdlog/spdlog.h>

Swapchain::Swapchain(const std::shared_ptr<VulkanContext>& context,
    const std::shared_ptr<Window>& window,
    bool immediate)
    : context(context), window(window), immediate(immediate)
{
    createSwapchain();
    createSwapchainImages();
}

Swapchain::~Swapchain()
{
    if (!context) return;
    destroySwapchainResources();
}

void Swapchain::destroySwapchainResources()
{
    VkDevice device = context->device;

    // 1) Semaphores
    for (auto s : imageAvailableSemaphores) {
        if (s != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, s, nullptr);
        }
    }
    imageAvailableSemaphores.clear();

    // 2) ImageViews
    for (auto& img : swapchainImages) {
        if (img && img->imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, img->imageView, nullptr);
            img->imageView = VK_NULL_HANDLE;
        }
    }
    swapchainImages.clear();

    // 3) Swapchain
    if (swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

void Swapchain::createSwapchain()
{
    VkPhysicalDevice physicalDevice = context->physicalDevice;
    VkSurfaceKHR surface = context->surface.value();

    // --- Query surface capabilities ---
    VkSurfaceCapabilitiesKHR capabilities{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    if (formatCount == 0) {
        throw std::runtime_error("No surface formats available");
    }
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    if (presentModeCount == 0) {
        throw std::runtime_error("No present modes available");
    }
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

    // --- Choose surface format ---
    surfaceFormat = formats[0];
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
        {
            surfaceFormat = f;
            break;
        }
    }
    swapchainFormat = surfaceFormat.format;
    spdlog::debug("Surface format: {}", string_VkFormat(swapchainFormat));

    // --- Choose present mode ---
    presentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto& pm : presentModes) {
        if (immediate && pm == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            presentMode = pm;
            break;
        }
        if (pm == VK_PRESENT_MODE_MAILBOX_KHR) {
            presentMode = pm;
            break;
        }
    }
    spdlog::debug("Present mode: {}", string_VkPresentModeKHR(presentMode));

    // --- Choose extent ---
    auto [width, height] = window->getFramebufferSize();
    VkExtent2D extent = capabilities.currentExtent;

    if (extent.width == UINT32_MAX || extent.width == 0) {
        extent.width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    swapchainExtent = extent;
    spdlog::debug("Swapchain extent chosen: {}x{}", extent.width, extent.height);

    // --- Choose image count ---
    imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    // --- Queue sharing ---
    std::vector<uint32_t> uniqueFamilies;
    uniqueFamilies.reserve(context->queues.size());

    for (auto& q : context->queues) {
        uint32_t fam = q.second.queueFamily;
        if (std::find(uniqueFamilies.begin(), uniqueFamilies.end(), fam) == uniqueFamilies.end())
            uniqueFamilies.push_back(fam);
    }

    VkSwapchainCreateInfoKHR create{};
    create.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create.surface = surface;
    create.minImageCount = imageCount;
    create.imageFormat = surfaceFormat.format;
    create.imageColorSpace = surfaceFormat.colorSpace;
    create.imageExtent = extent;
    create.imageArrayLayers = 1;

    create.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    if (uniqueFamilies.size() > 1) {
        create.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create.queueFamilyIndexCount = static_cast<uint32_t>(uniqueFamilies.size());
        create.pQueueFamilyIndices = uniqueFamilies.data();
    }
    else {
        create.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    create.preTransform = capabilities.currentTransform;
    create.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create.presentMode = presentMode;
    create.clipped = VK_TRUE;
    create.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(context->device, &create, nullptr, &swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain");
    }

    spdlog::debug("Swapchain created");
}

void Swapchain::createSwapchainImages()
{
    uint32_t count = 0;
    vkGetSwapchainImagesKHR(context->device, swapchain, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error("vkGetSwapchainImagesKHR returned 0 images");
    }

    std::vector<VkImage> images(count);
    vkGetSwapchainImagesKHR(context->device, swapchain, &count, images.data());

    swapchainImages.clear();
    swapchainImages.reserve(count);

    for (auto img : images) {
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = img;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = swapchainFormat;
        view.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
        };
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.baseMipLevel = 0;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount = 1;

        VkImageView imageView = VK_NULL_HANDLE;
        if (vkCreateImageView(context->device, &view, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swapchain imageView");
        }

        // Wrap into your Image class (swapchain VkImage is owned by Vulkan)
        swapchainImages.push_back(
            std::make_shared<Image>(img, imageView, swapchainFormat, swapchainExtent)
        );
    }

    // Create semaphores per swapchain image (match original behavior)
    imageAvailableSemaphores.resize(swapchainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo sem{};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (uint32_t i = 0; i < imageAvailableSemaphores.size(); i++) {
        VkResult r = vkCreateSemaphore(context->device, &sem, nullptr, &imageAvailableSemaphores[i]);
        if (r != VK_SUCCESS) {
            throw std::runtime_error("Failed to create imageAvailable semaphore");
        }
    }
}

void Swapchain::recreate()
{
    vkDeviceWaitIdle(context->device);

    destroySwapchainResources();

    createSwapchain();
    createSwapchainImages();

    spdlog::debug("Swapchain recreated");
}
