#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>

#include "frameworks/3dgs.h"
#include "frameworks/GLFWWindow.h"

int main() {
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");

    try {
        auto window = std::make_shared<GLFWWindow>("HybridWeave", 1920, 1080);

        // config renderer
        RendererConfiguration config;
		// if you have't set up validation layers, disable them here to avoid warnings
        config.enableVulkanValidationLayers = false;
		// if you want to see the detailed timing breakdown of each pass, enable the query manager (requires GPU with timestamp query support)
		config.enableQueryManager = false;
        config.physicalDeviceId = 0;
        config.immediateSwapchain = false;

		config.gltfRenderMode = 0;
		config.gsRenderMode = 0;   // 0 = VRS with out early-z, 1 = Mix-with early-z
		config.lightingMode = 0;
        config.presentMode = 2;    // 0 = only gltf, 1 = only 3DGS, 2 = mix

		// you can replace the scene files here to test with your own scenes, just make sure the camera settings in RendererConfiguration are appropriate for the scene scale
        config.sceneGS = "assets/pointclouds/bicycle.ply";
		// you can replace the glTF scene with your own model, but make sure it has a PBR material and is not too heavy for testing (we recommend using a single object with less than 100k triangles for testing, and you can use the camera settings in RendererConfiguration to adjust the view)
        config.sceneGLTF = "assets/models/tree/tree_small_02_4k.gltf";
		// you can replace the IBL environment map here, just make sure it's an HDR image and adjust the camera settings in RendererConfiguration if the scene is too dark or too bright with the new environment
        config.iblHdrPath = "assets/envs/overcast_soil_puresky_4k.hdr";
        
		config.light.lightDir = glm::vec4(0.5f, 1.0f, 0.5f, 1.0f);
		config.light.lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 3.0f);
        config.light.envParams = glm::vec4(0.2f, 1.0f, 9.0f, 0.0f);
        config.window = window;

        VulkanSplatting app(config);

        app.start();
    }
    catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return -1;
    }

    return 0;
}
