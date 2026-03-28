#pragma once

#include <optional>
#include <string>
#include <memory>

#include "RendererConfiguration.h"

class Window;
class Renderer;

class VulkanSplatting {
public:

    explicit VulkanSplatting(RendererConfiguration configuration)
        : configuration(std::move(configuration)) {
    }

    static std::shared_ptr<Window> createGlfwWindow(std::string name, int width, int height);

    void start();
    void initialize();
    void logTranslation(float x, float y);
    void logMovement(float x, float y, float z);
    void stop();

private:
    RendererConfiguration configuration;
    std::shared_ptr<Renderer> renderer;
};

