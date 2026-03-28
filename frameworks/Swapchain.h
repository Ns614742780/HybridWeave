#pragma once

#include <memory>
#include <vector>
#include <stdexcept>

#include "VulkanContext.h"
#include "Window.h"

class Swapchain {
public:
    Swapchain(const std::shared_ptr<VulkanContext>& context,
        const std::shared_ptr<Window>& window,
        bool immediate);
    ~Swapchain();

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D swapchainExtent{};
    std::vector<std::shared_ptr<Image>> swapchainImages;

    std::vector<VkSemaphore> imageAvailableSemaphores;

    VkSurfaceFormatKHR surfaceFormat{};
    VkFormat swapchainFormat{};
    VkPresentModeKHR presentMode{};
    uint32_t imageCount = 0;

    void recreate();

private:
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<Window> window;
    bool immediate = false;

    void createSwapchain();
    void createSwapchainImages();

    void destroySwapchainResources();
};
