#pragma once

#include <optional>
#include <string>
#include <memory>
#include <glm/glm.hpp>

class Window;

#define LOG_PATH "logs"

struct LightUBO {
    glm::vec4 lightDir;    // xyz direction (world), w unused
    glm::vec4 lightColor;  // rgb color, w intensity
    glm::vec4 envParams;    //  x = IBL intensity
                             // y = IBL enable (0/1)
                             // z = prefilter mipLevels
                             // w = reserved
};

struct RendererConfiguration {
	// vulkan settings
    bool enableVulkanValidationLayers = false;
	bool enableQueryManager = false;
    std::optional<uint8_t> physicalDeviceId = std::nullopt;
    bool immediateSwapchain = false;

    // renderer settings
	int gltfRenderMode = 0;
	int gsRenderMode = 0;   // 0 = VRS, 1 = Mix
	int lightingMode = 0;
	int presentMode = 0;    // 0 = gltf, 1 = 3DGS, 2 = mix

	// path to scene file
    std::string sceneGS;
    std::string sceneGLTF;

    // camera
    float fov = 45.0f;
    float nearPlane = 0.2f;
    float farPlane = 3000.0f;

	// lighting
    LightUBO light{
        glm::vec4(-0.5f, -1.0f, -0.5f, 1.0f), // light direction
        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),     // light color + intensity
        glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),    // env intensity + enable
	};
    std::string iblHdrPath;

    // window
    std::shared_ptr<Window> window;
};
