#pragma once

#define GLM_SWIZZLE

#include <atomic>
#include <memory>
#include <vector>
#include <deque>
#include <chrono>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Window.h"
#include "GSScene.h"
#include "ComputePipeline.h"
#include "Swapchain.h"
#include "QueryManager.h"
#include "Buffer.h"
#include "DescriptorSet.h"
#include "Camera.h"
#include "RendererConfiguration.h"
#include "RenderGlobalResources.h"
#include "IRenderPass.h"

class Renderer {
public:

    explicit Renderer(RendererConfiguration configuration);
    ~Renderer();

    void initialize();
    void run();
    void stop();

private:
    void draw();

    void initializeVulkan();
    void createSwapchain();
    void createUniformBuffer();
    void createSyncObjects();
    void createQueryResources();
    void createCommandPool();
    void createRenderCommandBuffer();
    void createPasses();

    void updateUniforms();

    void submitSingleFrame(uint32_t frameIndex);
    void presentSingleFrame(uint32_t frameIndex);

    void handleInput();

    void recreateSwapchain();

private:

    RendererConfiguration configuration;
    RenderGlobalResources globalResources;

    std::shared_ptr<Window>        window;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<Swapchain>     swapchain;
    std::shared_ptr<GSScene>       scene;
    std::shared_ptr<QueryManager>  queryManager;

    std::shared_ptr<Buffer> uniformBuffer;
    std::shared_ptr<Buffer> lightUboBuffer;

	// if you want to start with a different camera position, rotation or projection parameters, change the values here
    Camera camera{
        glm::vec3(3.600350f, -2.643269f, 5.880525f),
        glm::quat(-0.061977f, -0.266743f, 0.217666f, -0.936818f),
        45,
        0.1f,
        3000.0f
    };

    std::vector<std::unique_ptr<IRenderPass>> passes;

    VkCommandPool   commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer>   renderCommandBuffers;

    std::vector<VkFence>     inflightFences;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
 
    uint32_t currentImageIndex = 0;
    uint32_t currentFrame = 0;

    std::atomic<bool> running{ true };

    int fpsCounter = 0;
    std::chrono::high_resolution_clock::time_point lastFpsTime =
        std::chrono::high_resolution_clock::now();

    uint64_t avgFpsFrameCounter = 0;
    std::chrono::high_resolution_clock::time_point avgFpsStartTime;
	double avgFpsIntervalSeconds = 15.0; // calculate average FPS over 15 seconds


public:
    void addPass(std::unique_ptr<IRenderPass> pass)
    {
        passes.push_back(std::move(pass));
    }
	Camera& getCamera() { return camera; }
};
