#include "GLFWWindow.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdexcept>

GLFWWindow::GLFWWindow(std::string name, int width, int height)
{
    if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
}

GLFWWindow::~GLFWWindow()
{
    if (window) {
        glfwDestroyWindow(static_cast<GLFWwindow*>(window));
        window = nullptr;
    }
}

VkSurfaceKHR GLFWWindow::createSurface(std::shared_ptr<VulkanContext> context)
{
    if (surface != VK_NULL_HANDLE) return surface;

    if (glfwCreateWindowSurface(context->instance, static_cast<GLFWwindow*>(window), nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    return surface;
}

void GLFWWindow::setTitle(const std::string& title)
{
    if (!window) return;
    glfwSetWindowTitle(
        static_cast<GLFWwindow*>(window),
        title.c_str()
    );
}

std::array<bool, 3> GLFWWindow::getMouseButton()
{
    GLFWwindow* win = static_cast<GLFWwindow*>(window);
    return {
        glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS,
        glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS,
        glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
    };
}

std::vector<std::string> GLFWWindow::getRequiredInstanceExtensions()
{
    uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string> r;
    r.reserve(count);
    for (uint32_t i = 0; i < count; ++i) r.emplace_back(exts[i]);
    return r;
}

std::pair<uint32_t, uint32_t> GLFWWindow::getFramebufferSize() const
{
    int w = 0, h = 0;
    glfwGetFramebufferSize(static_cast<GLFWwindow*>(window), &w, &h);
    return { (uint32_t)w, (uint32_t)h };
}

std::array<double, 2> GLFWWindow::getCursorTranslation()
{
    GLFWwindow* win = static_cast<GLFWwindow*>(window);

    double x, y;
    glfwGetCursorPos(win, &x, &y);

    if (firstMouse) {
        lastX = x; lastY = y;
        firstMouse = false;
        return { 0.0, 0.0 };
    }

    std::array<double, 2> d{ x - lastX, y - lastY };
    lastX = x; lastY = y;
    return d;
}

std::array<bool, 7> GLFWWindow::getKeys()
{
    GLFWwindow* win = static_cast<GLFWwindow*>(window);
    return {
        glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS,
        glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS,
        glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS,
        glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS,
        glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS, 
        glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS,
        glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS
    };
}

void GLFWWindow::mouseCapture(bool capture)
{
    if (capture == captured) return;

    captured = capture;

    GLFWwindow* win = static_cast<GLFWwindow*>(window);
    glfwSetInputMode(win, GLFW_CURSOR, capture ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);

    firstMouse = true;
}

bool GLFWWindow::tick()
{
    glfwPollEvents();
    return !glfwWindowShouldClose(static_cast<GLFWwindow*>(window));
}
