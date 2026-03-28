#include "Renderer.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Shader.h"
#include "Utils.h"
#include "GsRenderPass.h"
#include "GsMixRenderPass.h"
#include "GltfRenderPass.h"
#include "IblEnvPass.h"
#include "LightingPass.h"
#include "PresentPass.h"
#include <GLFW/glfw3.h>

Renderer::Renderer(RendererConfiguration config)
    : configuration(std::move(config))
{
}

Renderer::~Renderer()
{
    if (context && context->device) {

        if (commandPool != VK_NULL_HANDLE) {
            for (auto c : renderCommandBuffers) {
                vkFreeCommandBuffers(context->device, commandPool, 1, &c);
            }
            vkDestroyCommandPool(context->device, commandPool, nullptr);
        }

        for (auto f : inflightFences) {
            vkDestroyFence(context->device, f, nullptr);
        }
        for (auto s : renderFinishedSemaphores) {
            vkDestroySemaphore(context->device, s, nullptr);
        }

        if (imageAvailableSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(context->device, imageAvailableSemaphore, nullptr);
        }
    }
}

void Renderer::initialize()
{
    initializeVulkan();
    createSwapchain();
	createUniformBuffer();
    createSyncObjects();
	createQueryResources();
    createCommandPool();
    createRenderCommandBuffer();

    globalResources.context = context;
    globalResources.swapchain = swapchain;
    globalResources.uniformBuffer = uniformBuffer;
    globalResources.lightUboBuffer = lightUboBuffer;
	globalResources.enableQueryManager = configuration.enableQueryManager;
    globalResources.queryManager = queryManager;
    globalResources.camera = &camera;
	globalResources.gazeData = std::make_shared<GazeData>();

    lightUboBuffer->upload(&configuration.light, sizeof(LightUBO));

    createPasses();
}

void Renderer::initializeVulkan()
{
    spdlog::debug("Initializing Vulkan");

    window = configuration.window;
    context = std::make_shared<VulkanContext>(
        window->getRequiredInstanceExtensions(),
        std::vector<std::string>{},
        configuration.enableVulkanValidationLayers
    );

    context->createInstance();

    VkSurfaceKHR surface = window->createSurface(context);
	context->surface = surface;

    context->selectPhysicalDevice(configuration.physicalDeviceId, surface);

    VkPhysicalDeviceFeatures pdf = {};
    pdf.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    pdf.shaderInt64 = VK_TRUE;

    VkPhysicalDeviceVulkan11Features pdf11 = {};
    pdf11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    VkPhysicalDeviceVulkan12Features pdf12 = {};
    pdf12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    pdf12.shaderBufferInt64Atomics = VK_TRUE;
    pdf12.shaderSharedInt64Atomics = VK_TRUE;

    context->createLogicalDevice(pdf, pdf11, pdf12);

    context->createDescriptorPool(1);
}

void Renderer::createSwapchain()
{
    spdlog::debug("Creating swapchain");
    swapchain = std::make_shared<Swapchain>(context, window, configuration.immediateSwapchain);
}

void Renderer::createUniformBuffer()
{
    uniformBuffer = Buffer::uniform(
        context,
        sizeof(UniformBuffer) * FRAMES_IN_FLIGHT
    );
    lightUboBuffer = Buffer::uniform(
        context, 
        sizeof(LightUBO)
    );
}

void Renderer::createSyncObjects() 
{
	spdlog::debug("Creating synchronization objects");

	//// clean up old sync objects if any
    for (VkFence f : inflightFences) {
        if (f != VK_NULL_HANDLE) {
            vkDestroyFence(context->device, f, nullptr);
        }
    }
    inflightFences.clear();

    for (VkSemaphore s : renderFinishedSemaphores) {
        if (s != VK_NULL_HANDLE) {
            vkDestroySemaphore(context->device, s, nullptr);
        }
    }
    renderFinishedSemaphores.clear();

    if (imageAvailableSemaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(context->device, imageAvailableSemaphore, nullptr);
        imageAvailableSemaphore = VK_NULL_HANDLE;
    }


	// fences : one per frame
    inflightFences.resize(FRAMES_IN_FLIGHT);
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
        if (vkCreateFence(context->device, &info, nullptr, &inflightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create fence");
    }

    // imageAvailable semaphore
    {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(context->device, &semInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS)
            throw std::runtime_error("Failed to create semaphore");
    }

	// per swapchain image renderFinished semaphores
    const size_t imageCount = swapchain->swapchainImages.size();
    if (imageCount == 0) {
        throw std::runtime_error("Swapchain has no images when creating sync objects");
    }

    renderFinishedSemaphores.resize(imageCount);

    for (int i = 0; i < imageCount; i++) {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(context->device, &semInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create semaphore");
    }
}

void Renderer::createQueryResources()
{
    constexpr uint32_t MAX_QUERIES_PER_FRAME = 512;

    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = MAX_QUERIES_PER_FRAME;

    if (vkCreateQueryPool(
        context->device,
        &info,
        nullptr,
        &context->queryPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create query pool");
    }

    queryManager = std::make_shared<QueryManager>(
        context->device,
        context->queryPool,
        MAX_QUERIES_PER_FRAME,
        FRAMES_IN_FLIGHT
    );
}


void Renderer::createCommandPool()
{
    spdlog::debug("Creating command pool");
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = context->queues[VulkanContext::Queue::GRAPHICS].queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(context->device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void Renderer::createRenderCommandBuffer() 
{
    renderCommandBuffers.resize(FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = FRAMES_IN_FLIGHT;

    if (vkAllocateCommandBuffers(context->device, &alloc, renderCommandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
}

void Renderer::createPasses()
{
    spdlog::debug("Creating render passes");
    passes.clear();

    IGBufferProvider* gbufferProvider = nullptr;
    IRenderPass* lightingRaw = nullptr;
    IRenderPass* gsRaw = nullptr;
    IblEnvPass* iblEnvPass = nullptr;

    if(configuration.presentMode == 0 || configuration.presentMode == 2) {
        if (configuration.gltfRenderMode == 0) {
            auto* gltfPass = new GltfRenderPass(globalResources, configuration.sceneGLTF);
            gbufferProvider = gltfPass;
            addPass(std::unique_ptr<IRenderPass>(gltfPass));
        }
        else {
            throw std::runtime_error("Invalid gltf mode");
        }

        iblEnvPass = new IblEnvPass(globalResources, configuration.iblHdrPath);
        addPass(std::unique_ptr<IRenderPass>(iblEnvPass));

        if (configuration.lightingMode == 0) {
            auto* p = new LightingPass(globalResources, gbufferProvider, iblEnvPass, -1);
            lightingRaw = p;
            addPass(std::unique_ptr<IRenderPass>(p));
        }
        else {
			throw std::runtime_error("Invalid lighting mode");
        }
	}

    if(configuration.presentMode == 1) {
        auto* gsPass = new GsRenderPass(globalResources, configuration.sceneGS);
        gsRaw = gsPass;
        gsPass->setOccluderDepthProvider(nullptr);
        addPass(std::unique_ptr<IRenderPass>(gsPass));
	}
    if (configuration.presentMode == 2) {
        if (configuration.gsRenderMode == 0) {
            auto* gsPass = new GsRenderPass(globalResources, configuration.sceneGS);
            gsRaw = gsPass;
            gsPass->setOccluderDepthProvider(gbufferProvider);
            addPass(std::unique_ptr<IRenderPass>(gsPass));
        }
        else if(configuration.gsRenderMode == 1) {
            auto* gsPass = new GsMixRenderPass(globalResources, configuration.sceneGS);
            gsRaw = gsPass;
            gsPass->setOccluderDepthProvider(gbufferProvider);
            addPass(std::unique_ptr<IRenderPass>(gsPass));
		}
	}

    if (configuration.presentMode == 0) {
		auto* gltfColorProvider = dynamic_cast<IColorProvider*>(lightingRaw);
        PresentPass::Params pp{};
        pp.mode = PresentPass::Mode::PresentA;

        addPass(std::unique_ptr<IRenderPass>(
            new PresentPass(globalResources, gltfColorProvider, nullptr, pp)
        ));
    }
    else if (configuration.presentMode == 1) {
		auto* gsColorProvider = dynamic_cast<IColorProvider*>(gsRaw);
        PresentPass::Params pp{};
        pp.mode = PresentPass::Mode::PresentB;

        addPass(std::unique_ptr<IRenderPass>(
            new PresentPass(globalResources, gsColorProvider, nullptr, pp)
        ));
    }
    else {
		auto* gltfColorProvider = dynamic_cast<IColorProvider*>(lightingRaw);
		auto* gsColorProvider = dynamic_cast<IColorProvider*>(gsRaw);
        PresentPass::Params pp{};
        pp.mode = PresentPass::Mode::Mix;
        pp.mixOp = 0;        // debug op
        pp.mixFactor = 0.85f; // mix strength
        pp.openMixEnhance = true;
		pp.enableBlurMap = 1; // enable blur map
		pp.styleLock = 1;   // style lock

        auto presentPass = new PresentPass(globalResources, gltfColorProvider, gsColorProvider, pp);
		presentPass->setGBufferProvider(gbufferProvider);
        addPass(std::unique_ptr<IRenderPass>(presentPass));
    }

    for (auto& pass : passes) {
        pass->initialize();
    }
}

void Renderer::handleInput()
{
    auto translation = window->getCursorTranslation();
    auto keys = window->getKeys();
    auto mouseButtons = window->getMouseButton();

	// left mouse button to rotate camera
    if (mouseButtons[0]) {
        window->mouseCapture(true);

        if (translation[0] != 0.0 || translation[1] != 0.0) {
            const float sensitivity = 0.005f;

            float dx = static_cast<float>(translation[0]);
            float dy = static_cast<float>(translation[1]);

			// 1️⃣ yaw: rotate around world's up (y) axis
            glm::quat qYaw = glm::angleAxis(
                -dx * sensitivity,
                glm::vec3(0.0f, 1.0f, 0.0f)
            );

			// 2️⃣ pitch: rotate around camera's right axis
            glm::vec3 right =
                glm::normalize(glm::mat3_cast(camera.rotation) * glm::vec3(1.0f, 0.0f, 0.0f));

            glm::quat qPitch = glm::angleAxis(
                +dy * sensitivity,
                right
            );

            camera.rotation = glm::normalize(qYaw * qPitch * camera.rotation);
        }
    }
    else {
        window->mouseCapture(false);
    }

	// WASD + move up/down keys
    glm::vec3 direction(0.0f);
    if (keys[0]) direction += glm::vec3(0.0f, 0.0f, -1.0f); // W
    if (keys[1]) direction += glm::vec3(-1.0f, 0.0f, 0.0f); // A
    if (keys[2]) direction += glm::vec3(0.0f, 0.0f, 1.0f);  // S
    if (keys[3]) direction += glm::vec3(1.0f, 0.0f, 0.0f);  // D
    if (keys[4]) direction += glm::vec3(0.0f, 1.0f, 0.0f);  // Q / up
    if (keys[5]) direction += glm::vec3(0.0f, -1.0f, 0.0f); // E / down

    if (direction != glm::vec3(0.0f)) {
        direction = glm::normalize(direction);

		float speed = 0.1f; // units per second
        glm::vec3 worldDir = glm::mat3_cast(camera.rotation) * direction;
        camera.position += worldDir * speed;
    }
}

void Renderer::draw()
{
    const uint32_t frameIndex = currentFrame % FRAMES_IN_FLIGHT;

    // 1) CPU wait for previous work
    vkWaitForFences(context->device, 1, &inflightFences[frameIndex], VK_TRUE, UINT64_MAX);
    vkResetFences(context->device, 1, &inflightFences[frameIndex]);
	// retrieve timestamps from previous frame
    if (queryManager) {
        queryManager->resolveAndPrint(frameIndex);
    }

	// update passes
    for (auto& pass : passes) {
        pass->update(0.0f);
    }
		
    // 2) Acquire image
    VkResult r = vkAcquireNextImageKHR(
        context->device,
        swapchain->swapchain,
        UINT64_MAX,
        imageAvailableSemaphore,
        VK_NULL_HANDLE,
        &currentImageIndex
    );

    switch (r)
    {
    case VK_SUCCESS:
        break;

    case VK_TIMEOUT:
    case VK_NOT_READY:
        return;
        break;

    case VK_SUBOPTIMAL_KHR:
    case VK_ERROR_OUT_OF_DATE_KHR:
        recreateSwapchain();
        return;

    default:
        throw std::runtime_error("Failed to acquire swap chain image");
    }

    // 3) Camera input + uniform update
    handleInput();
    updateUniforms();

    // 4) Build primary command buffer fresh every frame
	VkCommandBuffer cmd = renderCommandBuffers[frameIndex];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin);

    if (queryManager) {
        queryManager->beginFrame(frameIndex, cmd);

        if (globalResources.enableQueryManager && queryManager) {
            queryManager->writeTimestamp(
                frameIndex, cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                "frame_total_start");
        }
    }

    // 5) record passes
    if (globalResources.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            "passes_record_start");
    }
    for (auto& pass : passes)
        pass->record(cmd, currentImageIndex, frameIndex);
    if (globalResources.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            "passes_record_end");

        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            "frame_total_end");
    }

    vkEndCommandBuffer(cmd);

    // 6) Submit queue
    submitSingleFrame(frameIndex);
    // 7) Present
    presentSingleFrame(frameIndex);

    currentFrame = (currentFrame + 1) % FRAMES_IN_FLIGHT;
}

void Renderer::updateUniforms()
{
    UniformBuffer data{};
    auto [width, height] = swapchain->swapchainExtent;
    data.width = width;
    data.height = height;

    data.camera_position = glm::vec4(camera.position, 1.0f);

    auto rotation = glm::mat4_cast(camera.rotation);
    auto translation = glm::translate(glm::mat4(1.0f), camera.position);

    data.view_mat = glm::inverse(translation * rotation);

    const float tan_fovx = std::tan(glm::radians(camera.fov) * 0.5f);
    const float tan_fovy = tan_fovx * float(height) / float(width);

    const float fovy = 2.0f * std::atan(tan_fovy);

    glm::mat4 proj_gltf = glm::perspective(
        fovy,
        float(width) / float(height),
        camera.nearPlane,
        camera.farPlane
    );

    proj_gltf[0][1] *= -1.0f;
    proj_gltf[1][1] *= -1.0f;
    proj_gltf[2][1] *= -1.0f;
    proj_gltf[3][1] *= -1.0f;

    data.proj_mat = proj_gltf;

    // only use for 3dgs
    data.view_proj = glm::perspective(std::atan(tan_fovy) * 2.0f,
        static_cast<float>(width) / static_cast<float>(height),
        camera.nearPlane,
        camera.farPlane) * data.view_mat;
    data.view_proj[0][1] *= -1.0f;
    data.view_proj[1][1] *= -1.0f;
    data.view_proj[2][1] *= -1.0f;
    data.view_proj[3][1] *= -1.0f;

    data.tan_fovx = tan_fovx;
    data.tan_fovy = tan_fovy;

    // foveation params
    float minWH = float(std::min(width, height));
    glm::vec2 gazePx(width * 0.5f, height * 0.5f);

    float R0 = minWH * 0.15f;
    float R1 = minWH * 0.35f;
    data.gaze_params = glm::vec4(gazePx, R0, R1);
    globalResources.gazeData->gazeParams = data.gaze_params;

    uniformBuffer->upload(&data, sizeof(UniformBuffer), 0);
}

void Renderer::submitSingleFrame(uint32_t frameIndex)
{
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailableSemaphore;
    submit.pWaitDstStageMask = &waitStage;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &renderCommandBuffers[frameIndex];

    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &renderFinishedSemaphores[currentImageIndex];

    vkQueueSubmit(
        context->queues[VulkanContext::Queue::GRAPHICS].queue,
        1,
        &submit,
        inflightFences[frameIndex]
    );
}

void Renderer::presentSingleFrame(uint32_t frameIndex)
{
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinishedSemaphores[currentImageIndex];

    present.swapchainCount = 1;
    present.pSwapchains = &swapchain->swapchain;
    present.pImageIndices = &currentImageIndex;

    VkResult r = vkQueuePresentKHR(
        context->queues[VulkanContext::Queue::PRESENT].queue,
        &present
    );

    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        recreateSwapchain();
    else if (r != VK_SUCCESS)
        throw std::runtime_error("Failed to present swapchain image");
}

void Renderer::run()
{
    lastFpsTime = std::chrono::high_resolution_clock::now();
    avgFpsStartTime = lastFpsTime;

    fpsCounter = 0;
    avgFpsFrameCounter = 0;

    float lastInstantFps = 0.0f;
    float lastAvgFps = 0.0f;

    while (running) {
        if (!window->tick()) {
            break;
        }

        draw();

        fpsCounter++;
        avgFpsFrameCounter++;

        auto now = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastFpsTime).count();

        std::string title = "FPS:";
        if (diff >= 1000) {
            lastInstantFps = fpsCounter * 1000.0f / diff;

            fpsCounter = 0;
            lastFpsTime = now;
        }

        double avgElapsed =
            std::chrono::duration<double>(now - avgFpsStartTime).count();

        if (avgElapsed >= avgFpsIntervalSeconds) {
            lastAvgFps = avgFpsFrameCounter / avgElapsed;

            avgFpsFrameCounter = 0;
            avgFpsStartTime = now;
        }

        {
            std::string title =
                "Mixed Renderer | FPS: " +
                std::to_string(lastInstantFps);

            title += " | AVG FPS: " + std::to_string(lastAvgFps);

            CameraSnapshot snapshot;
			snapshot.position = camera.position;
			snapshot.rotation = camera.rotation;
			snapshot.fov_y_deg = camera.fov;
			snapshot.near_plane = camera.nearPlane;
			snapshot.far_plane = camera.farPlane;
			auto [width, height] = swapchain->swapchainExtent;
			snapshot.aspect = static_cast<float>(width) / static_cast<float>(height);

			appendCameraLog(snapshot, "camera.txt", lastAvgFps);

            window->setTitle(title);
        }
    }

    vkDeviceWaitIdle(context->device);
}

void Renderer::stop()
{
    running = false;
    if (context && context->device) {
        vkDeviceWaitIdle(context->device);
    }
}

void Renderer::recreateSwapchain()
{
    spdlog::debug("Recreating swapchain");
    
    vkDeviceWaitIdle(context->device);
    
    auto oldExtent = swapchain->swapchainExtent;
    swapchain->recreate();

    if (swapchain->swapchainExtent.width == oldExtent.width &&
        swapchain->swapchainExtent.height == oldExtent.height) {
        return;
    }

    for (auto& pass : passes) {
        if (pass) {
            pass->onSwapchainResized();
        }
    }
}