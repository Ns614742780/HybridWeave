#include "3dgs.h"
#include "Renderer.h"
#include "GLFWWindow.h"
#include <spdlog/spdlog.h>

std::shared_ptr<Window> VulkanSplatting::createGlfwWindow(
    std::string name, int width, int height)
{
    return std::make_shared<GLFWWindow>(name, width, height);
}


void VulkanSplatting::initialize() {
    renderer = std::make_shared<Renderer>(configuration);
    renderer->initialize();
}


void VulkanSplatting::start() {
    renderer = std::make_shared<Renderer>(configuration);
    renderer->initialize();
    renderer->run();
}


void VulkanSplatting::logTranslation(float x, float y) {
    if (configuration.window)
        configuration.window->logTranslation(x, y);
}


void VulkanSplatting::logMovement(float x, float y, float z) {
    if (renderer)
        renderer->getCamera().translate(glm::vec3(x, y, z));
}


void VulkanSplatting::stop() {
    if (renderer)
        renderer->stop();
}
