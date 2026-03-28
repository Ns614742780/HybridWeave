#pragma once
#include "Window.h"
#include <array>
#include <vector>
#include <string>

class GLFWWindow final : public Window {
public:
    GLFWWindow(std::string name, int width, int height);
    ~GLFWWindow() override;

    VkSurfaceKHR createSurface(std::shared_ptr<VulkanContext> context) override;
    void setTitle(const std::string& title) override;

    std::array<bool, 3> getMouseButton() override;
    std::vector<std::string> getRequiredInstanceExtensions() override;
    std::pair<uint32_t, uint32_t> getFramebufferSize() const override;

    std::array<double, 2> getCursorTranslation() override;
    std::array<bool, 7> getKeys() override;

    void mouseCapture(bool capture) override;
    bool tick() override;

    void* window = nullptr;

private:
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    bool firstMouse = true;
    double lastX = 0.0;
    double lastY = 0.0;

    bool captured = false;
};
