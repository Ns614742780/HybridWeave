#include "GsMixRenderPass.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Shader.h"
#include "Utils.h"
#include "VulkanContext.h"
#include "Swapchain.h"
#include "Buffer.h"
#include "Image.h"
#include "QueryManager.h"
#include "Camera.h"
#include "GSScene.h"
#include "DescriptorSet.h"
#include "ComputePipeline.h"
#include "IGBufferProvider.h"
#include <GLFW/glfw3.h>

static void InferSrcStageAccess(VkImageLayout oldLayout,
    VkPipelineStageFlags& srcStage,
    VkAccessFlags& srcAccess)
{
    srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    srcAccess = 0;

    if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        srcAccess = VK_ACCESS_SHADER_READ_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL) {
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        srcAccess = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
    }
}

static glm::mat4 ComputeProjGltf_FromCameraLikeRenderer(const Camera& cam, uint32_t width, uint32_t height)
{
    const float tan_fovx = std::tan(glm::radians(cam.fov) * 0.5f);
    const float tan_fovy = tan_fovx * float(height) / float(width);
    const float fovy = 2.0f * std::atan(tan_fovy);

    glm::mat4 proj = glm::perspective(
        fovy,
        float(width) / float(height),
        cam.nearPlane,
        cam.farPlane
    );

    proj[0][1] *= -1.0f;
    proj[1][1] *= -1.0f;
    proj[2][1] *= -1.0f;
    proj[3][1] *= -1.0f;

    return proj;
}

GsMixRenderPass::GsMixRenderPass(
    const RenderGlobalResources& r,
    const std::string& scenePath
)
	: global(r)
    , context(r.context)
    , swapchain(r.swapchain)
    , uniformBuffer(r.uniformBuffer)
    , queryManager(r.queryManager)
    , camera(r.camera)
{
    scene = std::make_shared<GSScene>(scenePath);

    if (!context)
        throw std::runtime_error("GsMixRenderPass: VulkanContext missing (RenderGlobalResources wrong)");
    if (!swapchain)
        throw std::runtime_error("GsMixRenderPass: Swapchain missing (RenderGlobalResources wrong)");
    if (!uniformBuffer)
        throw std::runtime_error("GsMixRenderPass: uniformBuffer missing (RenderGlobalResources wrong)");
    if (!camera)
        throw std::runtime_error("GsMixRenderPass: camera pointer missing (RenderGlobalResources wrong)");
    if (!scene)
        throw std::runtime_error("GsMixRenderPass: GSScene missing");
}

GsMixRenderPass::~GsMixRenderPass()
{
    vkDeviceWaitIdle(context->device);

    destroyMeshDepthSampler();
    destroyMeshTileImages();
    destroyGsOffscreenImages();
}

void GsMixRenderPass::onSwapchainResized()
{
    rebuildForResolutionChange();
}

void GsMixRenderPass::initialize()
{
    loadScene();
    createBuffers();

    createPreprocessPipeline();
    createPreprocessPipelineMix();
    createMeshDepthTilePipeline();

    createPrefixSumPipeline();
    createRadixSortPipeline();
    createPreprocessSortPipeline();
    createTileBoundaryPipeline();

    rebuildForResolutionChange();
}

void GsMixRenderPass::update(float dt)
{
    if (!hasTotalSumReadback) {
        return;
    }

    cachedNumInstances = totalSumBufferHost->readOne<uint32_t>();
    ensureSortCapacity(cachedNumInstances);

    static uint64_t c = 0;
    if ((c++ % 30) == 0) {
        spdlog::info("[GsMixRenderPass] totalSum(readback)={}", cachedNumInstances);
    }
}

void GsMixRenderPass::record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex)
{
    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            "gs_mix_total_start");
    }
    if (occluder && meshTilePipeline && meshTileInputSet && meshTileOutputSet) {
        dispatchMeshDepthTile(cmd, imageIndex, frameIndex);
    }

    dispatchPreprocess(cmd, imageIndex, frameIndex);
    dispatchPrefixSum(cmd, frameIndex);

    hasTotalSumReadback = true;

    vertexAttributeBuffer->computeWriteReadBarrier(cmd);

    dispatchPreprocessSort(cmd, frameIndex);
    dispatchRadixSort(cmd, frameIndex);
    dispatchTileBoundary(cmd, frameIndex);
    dispatchRender(cmd, imageIndex, frameIndex);

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            "gs_mix_total_end");
    }
}

void GsMixRenderPass::loadScene()
{
    spdlog::debug("Loading scene to GPU");
    scene->load(context);

    gaussianCount = static_cast<uint32_t>(scene->getNumVertices());

    if (gaussianCount == 0)
        throw std::runtime_error("GSScene: empty");
}

void GsMixRenderPass::createBuffers()
{
    const uint32_t N = static_cast<uint32_t>(scene->getNumVertices());
    gaussianCount = N;

    if (N == 0) {
        throw std::runtime_error("GsMixRenderPass::createBuffers: scene has 0 vertices");
    }

    vertexAttributeBuffer = Buffer::storage(
        context,
        static_cast<uint64_t>(N) * sizeof(VertexAttributeBuffer)
    );

    tileOverlapBuffer = Buffer::storage(
        context,
        static_cast<uint64_t>(N) * sizeof(uint32_t)
    );

    prefixSumPingBuffer = Buffer::storage(
        context,
        static_cast<uint64_t>(N) * sizeof(uint32_t)
    );

    prefixSumPongBuffer = Buffer::storage(
        context,
        static_cast<uint64_t>(N) * sizeof(uint32_t)
    );

    totalSumBufferHost = Buffer::staging(
        context,
        sizeof(uint32_t)
    );
    uint32_t zero = 0;
    totalSumBufferHost->upload(&zero, sizeof(uint32_t));

    const uint64_t sortElementCount =
        static_cast<uint64_t>(N) * static_cast<uint64_t>(sortBufferSizeMultiplier);

    sortKBufferEven = Buffer::storage(
        context,
        sortElementCount * sizeof(uint64_t),
        false, 0, "sortKBufferEven"
    );

    sortKBufferOdd = Buffer::storage(
        context,
        sortElementCount * sizeof(uint64_t),
        false, 0, "sortKBufferOdd"
    );

    sortVBufferEven = Buffer::storage(
        context,
        sortElementCount * sizeof(uint32_t),
        false, 0, "sortVBufferEven"
    );

    sortVBufferOdd = Buffer::storage(
        context,
        sortElementCount * sizeof(uint32_t),
        false, 0, "sortVBufferOdd"
    );

    uint32_t globalInvocationSize =
        static_cast<uint32_t>(sortElementCount / numRadixSortBlocksPerWorkgroup);

    const uint32_t remainder =
        static_cast<uint32_t>(sortElementCount % numRadixSortBlocksPerWorkgroup);

    if (remainder > 0) {
        globalInvocationSize++;
    }

    const uint32_t numWorkgroups = (globalInvocationSize + 256 - 1) / 256;

    sortHistBuffer = Buffer::storage(
        context,
        static_cast<uint64_t>(numWorkgroups) * 256u * sizeof(uint32_t),
        false,
        0,
        "sortHistBuffer"
    );
}

void GsMixRenderPass::createMeshTileImages()
{
    destroyMeshTileImages();

    VkExtent2D extent = swapchain->swapchainExtent;
    constexpr uint32_t TILE_SIZE = 16;

    meshTileExtent.width = (extent.width + TILE_SIZE - 1) / TILE_SIZE;
    meshTileExtent.height = (extent.height + TILE_SIZE - 1) / TILE_SIZE;

    const size_t imageCount = swapchain->swapchainImages.size();
    meshTileDepth.resize(imageCount);
    meshTileDepthLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);

    auto makeImage2D = [&](VkFormat fmt, VkExtent2D ex) -> std::shared_ptr<Image> {
        VkImageCreateInfo img{};
        img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img.imageType = VK_IMAGE_TYPE_2D;
        img.extent = { ex.width, ex.height, 1 };
        img.mipLevels = 1;
        img.arrayLayers = 1;
        img.format = fmt;
        img.tiling = VK_IMAGE_TILING_OPTIMAL;
        img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        img.samples = VK_SAMPLE_COUNT_1_BIT;
        img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        if (vmaCreateImage(context->allocator, &img, &alloc, &image, &allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("GsMixRenderPass: failed to create meshTileDepth image");

        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = img.format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.baseMipLevel = 0;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount = 1;

        VkImageView iv = VK_NULL_HANDLE;
        if (vkCreateImageView(context->device, &view, nullptr, &iv) != VK_SUCCESS) {
            vmaDestroyImage(context->allocator, image, allocation);
            throw std::runtime_error("GsMixRenderPass: failed to create meshTileDepth view");
        }

        auto out = std::make_shared<Image>();
        out->image = image;
        out->imageView = iv;
        out->mip0View = iv;
        out->format = img.format;
        out->extent = ex;
        out->framebuffer = VK_NULL_HANDLE;
        out->allocation = allocation;
        return out;
        };

    for (size_t i = 0; i < imageCount; ++i) {
        meshTileDepth[i] = makeImage2D(meshTileDepthFormat, meshTileExtent);
    }
}

void GsMixRenderPass::destroyMeshTileImages()
{
    auto destroyImage = [&](const std::shared_ptr<Image>& img) {
        if (!img) return;

        VkImageView iv = img->imageView;
        VkImageView mv = img->mip0View;

        if (mv != VK_NULL_HANDLE && mv != iv) {
            vkDestroyImageView(context->device, mv, nullptr);
        }
        if (iv != VK_NULL_HANDLE) {
            vkDestroyImageView(context->device, iv, nullptr);
        }

        if (img->image != VK_NULL_HANDLE && img->allocation != VK_NULL_HANDLE) {
            vmaDestroyImage(context->allocator, img->image, img->allocation);
        }

        img->mip0View = VK_NULL_HANDLE;
        img->imageView = VK_NULL_HANDLE;
        img->image = VK_NULL_HANDLE;
        img->allocation = VK_NULL_HANDLE;
        };

    for (auto& img : meshTileDepth) destroyImage(img);
    meshTileDepth.clear();
    meshTileDepthLayouts.clear();
    meshTileExtent = { 0,0 };
}


void GsMixRenderPass::createPreprocessPipeline()
{
    spdlog::debug("Creating preprocess pipeline");

    preprocessPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "preprocess"));

    preprocessInputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    preprocessInputSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, scene->vertexBuffer);
    preprocessInputSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, scene->cov3DBuffer);
    preprocessInputSet->build();
    preprocessPipeline->addDescriptorSet(0, preprocessInputSet);

    preprocessOutputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    preprocessOutputSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, uniformBuffer);
    preprocessOutputSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, vertexAttributeBuffer);
    preprocessOutputSet->bindBufferToDescriptorSet(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, tileOverlapBuffer);
    preprocessOutputSet->build();

    preprocessPipeline->addDescriptorSet(1, preprocessOutputSet);
    preprocessPipeline->build();
}


void GsMixRenderPass::createPreprocessPipelineMix()
{
    spdlog::debug("Creating preprocess MIX pipeline (preprocess_mix.comp + mesh tile depth)");

    preprocessPipelineMix = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "preprocess_mix"));

    if (!preprocessInputSet || !preprocessOutputSet) {
        throw std::runtime_error("GsMixRenderPass::createPreprocessPipelineMix: preprocessInputSet/outputSet not ready");
    }

    preprocessMixDepthSet = std::make_shared<DescriptorSet>(context, (uint32_t)swapchain->swapchainImages.size());

}

void GsMixRenderPass::createMeshDepthTilePipeline()
{
    spdlog::debug("Creating mesh tile depth pipeline (mesh_depth_tile.comp)");

    meshTilePipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "mesh_depth_tile"));

}


void GsMixRenderPass::createPrefixSumPipeline()
{
    spdlog::debug("Creating prefix sum pipeline");
    prefixSumPingBuffer = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t));
    prefixSumPongBuffer = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t));
    totalSumBufferHost = Buffer::staging(context, sizeof(uint32_t));

    prefixSumPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "prefix_sum"));

    prefixSumSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    prefixSumSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, prefixSumPingBuffer);
    prefixSumSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, prefixSumPongBuffer);
    prefixSumSet->build();

    prefixSumPipeline->addDescriptorSet(0, prefixSumSet);
    prefixSumPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t));
    prefixSumPipeline->build();
}

void GsMixRenderPass::createRadixSortPipeline()
{
    spdlog::debug("Creating radix sort pipeline");
    sortKBufferEven = Buffer::storage(context, scene->getNumVertices() * sizeof(uint64_t) * sortBufferSizeMultiplier, false, 0, "sortKBufferEven");
    sortKBufferOdd = Buffer::storage(context, scene->getNumVertices() * sizeof(uint64_t) * sortBufferSizeMultiplier, false, 0, "sortKBufferOdd");
    sortVBufferEven = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, false, 0, "sortVBufferEven");
    sortVBufferOdd = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, false, 0, "sortVBufferOdd");

    uint32_t globalInvocationSize =
        scene->getNumVertices() * sortBufferSizeMultiplier / numRadixSortBlocksPerWorkgroup;
    uint32_t remainder =
        scene->getNumVertices() * sortBufferSizeMultiplier % numRadixSortBlocksPerWorkgroup;
    if (remainder > 0) globalInvocationSize++;

    auto numWorkgroups = (globalInvocationSize + 256 - 1) / 256;
    sortHistBuffer = Buffer::storage(context, numWorkgroups * 256 * sizeof(uint32_t), false);

    sortHistPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "hist"));
    sortPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "sort"));

    radixHistSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    radixHistSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferEven);
    radixHistSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferOdd);
    radixHistSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortHistBuffer);
    radixHistSet->build();

    sortHistPipeline->addDescriptorSet(0, radixHistSet);
    sortHistPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstants));
    sortHistPipeline->build();

    radixSortSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    radixSortSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferEven);
    radixSortSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferOdd);
    radixSortSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferOdd);
    radixSortSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferEven);
    radixSortSet->bindBufferToDescriptorSet(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortVBufferEven);
    radixSortSet->bindBufferToDescriptorSet(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortVBufferOdd);
    radixSortSet->bindBufferToDescriptorSet(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortVBufferOdd);
    radixSortSet->bindBufferToDescriptorSet(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortVBufferEven);
    radixSortSet->bindBufferToDescriptorSet(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortHistBuffer);
    radixSortSet->build();

    sortPipeline->addDescriptorSet(0, radixSortSet);
    sortPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstants));
    sortPipeline->build();
}

void GsMixRenderPass::createPreprocessSortPipeline()
{
    spdlog::debug("Creating preprocess sort pipeline");
    preprocessSortPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "preprocess_sort"));

    preprocessSortSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    preprocessSortSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, vertexAttributeBuffer);
    preprocessSortSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, prefixSumPingBuffer);
    preprocessSortSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, prefixSumPongBuffer);
    preprocessSortSet->bindBufferToDescriptorSet(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferEven);
    preprocessSortSet->bindBufferToDescriptorSet(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortVBufferEven);
    preprocessSortSet->build();

    preprocessSortPipeline->addDescriptorSet(0, preprocessSortSet);
    preprocessSortPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t));
    preprocessSortPipeline->build();
}

void GsMixRenderPass::createTileBoundaryPipeline()
{
    spdlog::debug("Creating tile boundary pipeline");
    auto [width, height] = swapchain->swapchainExtent;
    auto tileX = (width + 16 - 1) / 16;
    auto tileY = (height + 16 - 1) / 16;
    tileBoundaryBuffer = Buffer::storage(context, tileX * tileY * sizeof(uint32_t) * 2);

    tileBoundaryPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "tile_boundary"));

    tileBoundarySet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    tileBoundarySet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortKBufferEven);
    tileBoundarySet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, tileBoundaryBuffer);
    tileBoundarySet->build();

    tileBoundaryPipeline->addDescriptorSet(0, tileBoundarySet);
    tileBoundaryPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t));
    tileBoundaryPipeline->build();
}

void GsMixRenderPass::createRenderPipeline()
{
    spdlog::debug("Creating render pipeline");
    renderPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "render"));

    renderInputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    renderInputSet->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, vertexAttributeBuffer);
    renderInputSet->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, tileBoundaryBuffer);
    renderInputSet->bindBufferToDescriptorSet(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, sortVBufferEven);
    renderInputSet->build();

    renderOutputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    for (auto& img : gsColor) {
        renderOutputSet->bindImageToDescriptorSet(
            0,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_COMPUTE_BIT,
            img
        );
    }
    for (auto& img : gsDepth) {
        renderOutputSet->bindImageToDescriptorSet(
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_COMPUTE_BIT,
            img
        );
    }
    renderOutputSet->build();

    renderPipeline->addDescriptorSet(0, renderInputSet);
    renderPipeline->addDescriptorSet(1, renderOutputSet);
    renderPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2);
    renderPipeline->build();
}

void GsMixRenderPass::createRenderPipelineMix()
{
    renderPipelineMix.reset();
    renderDepthSet.reset();

    if (!occluder) {
        return;
    }

    createMeshDepthSampler();

    spdlog::debug("Creating render MIX pipeline (render_mix.comp + mesh depth)");

    renderPipelineMix = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "render_mix"));

    if (!renderInputSet || !renderOutputSet) {
        throw std::runtime_error("GsMixRenderPass::createRenderPipelineMix: renderInputSet/renderOutputSet not ready");
    }

    const uint32_t imageCount = (uint32_t)swapchain->swapchainImages.size();
    const auto& gbufs = occluder->getGBufferImages();
    if ((uint32_t)gbufs.size() < imageCount) {
        throw std::runtime_error("GsMixRenderPass::createRenderPipelineMix: occluder gbuffer image count mismatch");
    }

    renderDepthSet = std::make_shared<DescriptorSet>(context, imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        renderDepthSet->bindCombinedImageSamplerToDescriptorSet(
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            gbufs[i].depth,
            meshDepthSampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    renderDepthSet->build();

    renderPipelineMix->addDescriptorSet(0, renderInputSet);
    renderPipelineMix->addDescriptorSet(1, renderOutputSet);
    renderPipelineMix->addDescriptorSet(2, renderDepthSet);

    renderPipelineMix->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MixPC));

    renderPipelineMix->build();
}

void GsMixRenderPass::createMeshDepthSampler()
{
    if (meshDepthSampler != VK_NULL_HANDLE) return;

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;
    si.anisotropyEnable = VK_FALSE;

    if (vkCreateSampler(context->device, &si, nullptr, &meshDepthSampler) != VK_SUCCESS) {
        throw std::runtime_error("GsMixRenderPass: failed to create meshDepthSampler");
    }
}

void GsMixRenderPass::destroyMeshDepthSampler()
{
    if (meshDepthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, meshDepthSampler, nullptr);
        meshDepthSampler = VK_NULL_HANDLE;
    }
}

void GsMixRenderPass::createGsOffscreenImages()
{
    destroyGsOffscreenImages();

    VkExtent2D extent = swapchain->swapchainExtent;
    const size_t imageCount = swapchain->swapchainImages.size();

    auto makeImage = [&](VkFormat fmt, VkExtent2D ex, uint32_t layers, VkImageViewType viewType) -> std::shared_ptr<Image> {
        VkImageCreateInfo img{};
        img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img.imageType = VK_IMAGE_TYPE_2D;
        img.extent = { ex.width, ex.height, 1 };
        img.mipLevels = 1;
        img.arrayLayers = layers;
        img.format = fmt;
        img.tiling = VK_IMAGE_TILING_OPTIMAL;
        img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        img.samples = VK_SAMPLE_COUNT_1_BIT;
        img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        if (vmaCreateImage(context->allocator, &img, &alloc, &image, &allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("GsMixRenderPass: failed to create offscreen image");

        auto makeView = [&](VkImageViewType vt) -> VkImageView {
            VkImageViewCreateInfo view{};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = image;
            view.viewType = vt;
            view.format = img.format;
            view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view.subresourceRange.baseMipLevel = 0;
            view.subresourceRange.levelCount = 1;
            view.subresourceRange.baseArrayLayer = 0;
            view.subresourceRange.layerCount = layers;

            VkImageView iv = VK_NULL_HANDLE;
            if (vkCreateImageView(context->device, &view, nullptr, &iv) != VK_SUCCESS)
                throw std::runtime_error("GsMixRenderPass: failed to create offscreen image view");
            return iv;
            };

        VkImageView fullView = VK_NULL_HANDLE;
        try {
            fullView = makeView(viewType);
        }
        catch (...) {
            if (fullView) vkDestroyImageView(context->device, fullView, nullptr);
            vmaDestroyImage(context->allocator, image, allocation);
            throw;
        }

        auto out = std::make_shared<Image>();
        out->image = image;
        out->imageView = fullView;
        out->mip0View = fullView;
        out->format = img.format;
        out->extent = ex;
        out->framebuffer = VK_NULL_HANDLE;
        out->allocation = allocation;
        return out;
        };

    auto makeImage2D = [&](VkFormat fmt) -> std::shared_ptr<Image> {
        return makeImage(fmt, extent, 1u, VK_IMAGE_VIEW_TYPE_2D);
        };

    gsColor.resize(imageCount);
    gsColorLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);

    gsDepth.resize(imageCount);
    gsDepthLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);

    for (size_t i = 0; i < imageCount; ++i) {
        gsColor[i] = makeImage2D(gsColorFormat);
        gsDepth[i] = makeImage2D(gsDepthFormat);
    }
}

void GsMixRenderPass::destroyGsOffscreenImages()
{
    auto destroyImage = [&](const std::shared_ptr<Image>& img) {
        if (!img) return;

        VkImageView iv = img->imageView;
        VkImageView mv = img->mip0View;

        if (mv != VK_NULL_HANDLE && mv != iv) {
            vkDestroyImageView(context->device, mv, nullptr);
        }
        if (iv != VK_NULL_HANDLE) {
            vkDestroyImageView(context->device, iv, nullptr);
        }

        if (img->image != VK_NULL_HANDLE && img->allocation != VK_NULL_HANDLE) {
            vmaDestroyImage(context->allocator, img->image, img->allocation);
        }

        img->mip0View = VK_NULL_HANDLE;
        img->imageView = VK_NULL_HANDLE;
        img->image = VK_NULL_HANDLE;
        img->allocation = VK_NULL_HANDLE;
        };

    for (auto& img : gsColor) destroyImage(img);
    gsColor.clear();
    gsColorLayouts.clear();

    for (auto& img : gsDepth) destroyImage(img);
    gsDepth.clear();
    gsDepthLayouts.clear();
}

void GsMixRenderPass::rebuildForResolutionChange()
{
    auto [width, height] = swapchain->swapchainExtent;
    gsExtent = swapchain->swapchainExtent;

    constexpr uint32_t TILE_SIZE = 16;

    uint32_t tileCountX = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tileCountY = (height + TILE_SIZE - 1) / TILE_SIZE;

    const uint64_t numTiles = static_cast<uint64_t>(tileCountX) * tileCountY;
    const uint64_t numUInts = numTiles * 2u;
    const VkDeviceSize bufferSize = numUInts * sizeof(uint32_t);

    if (!tileBoundaryBuffer) {
        tileBoundaryBuffer = Buffer::storage(context, bufferSize, false, 0, "TileBoundaryBuffer");
    }
    else {
        tileBoundaryBuffer->realloc(bufferSize);
    }

    createMeshTileImages();

    createGsOffscreenImages();
    createRenderPipeline();
    createRenderPipelineMix();

    if (occluder) {
        createMeshDepthSampler();

        const uint32_t imageCount = (uint32_t)swapchain->swapchainImages.size();
        const auto& gbufs = occluder->getGBufferImages();
        if ((uint32_t)gbufs.size() < imageCount) {
            throw std::runtime_error("GsMixRenderPass::rebuildForResolutionChange: occluder gbuffer count mismatch");
        }

        meshTileInputSet = std::make_shared<DescriptorSet>(context, imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            meshTileInputSet->bindCombinedImageSamplerToDescriptorSet(
                0,
                VK_SHADER_STAGE_COMPUTE_BIT,
                gbufs[i].depth,
                meshDepthSampler,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
        }
        meshTileInputSet->build();

        meshTileOutputSet = std::make_shared<DescriptorSet>(context, imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            meshTileOutputSet->bindImageToDescriptorSet(
                0,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_SHADER_STAGE_COMPUTE_BIT,
                meshTileDepth[i]
            );
        }
        meshTileOutputSet->build();

        meshTilePipeline.reset();
        meshTilePipeline = std::make_shared<ComputePipeline>(
            context, std::make_shared<Shader>(context, "mesh_depth_tile"));
        meshTilePipeline->addDescriptorSet(0, meshTileInputSet);
        meshTilePipeline->addDescriptorSet(1, meshTileOutputSet);
        meshTilePipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshTilePC));
        meshTilePipeline->build();

        preprocessPipelineMix = std::make_shared<ComputePipeline>(
            context, std::make_shared<Shader>(context, "preprocess_mix"));

        preprocessMixDepthSet = std::make_shared<DescriptorSet>(context, imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            preprocessMixDepthSet->bindCombinedImageSamplerToDescriptorSet(
                0,
                VK_SHADER_STAGE_COMPUTE_BIT,
                meshTileDepth[i],
                meshDepthSampler,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
        }
        preprocessMixDepthSet->build();

        preprocessPipelineMix->addDescriptorSet(0, preprocessInputSet);
        preprocessPipelineMix->addDescriptorSet(1, preprocessOutputSet);
        preprocessPipelineMix->addDescriptorSet(2, preprocessMixDepthSet);

        preprocessPipelineMix->build();
    }

    const size_t imageCount = swapchain->swapchainImages.size();
    swapImageLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);
}

void GsMixRenderPass::dispatchMeshDepthTile(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex)
{
    if (!occluder || !meshTilePipeline || !meshTileInputSet || !meshTileOutputSet) return;
    if (meshTileDepth.empty() || !meshTileDepth[imageIndex]) return;

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "gs_mesh_tile_start");
    }

    uint8_t curFrame = static_cast<uint8_t>(frameIndex % FRAMES_IN_FLIGHT);

    meshTilePipeline->bind(
        cmd,
        curFrame,
        Pipeline::DescriptorOption({ 0u, imageIndex })
    );

    {
        VkImage img = meshTileDepth[imageIndex]->image;

        VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags        srcAccess = 0;
        InferSrcStageAccess(meshTileDepthLayouts[imageIndex], srcStage, srcAccess);

        VkImageMemoryBarrier toGeneral{};
        toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toGeneral.oldLayout = meshTileDepthLayouts[imageIndex];
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = img;
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.baseMipLevel = 0;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.baseArrayLayer = 0;
        toGeneral.subresourceRange.layerCount = 1;
        toGeneral.srcAccessMask = srcAccess;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &toGeneral
        );

        meshTileDepthLayouts[imageIndex] = VK_IMAGE_LAYOUT_GENERAL;
    }

    MeshTilePC pc{};
    pc.width = swapchain->swapchainExtent.width;
    pc.height = swapchain->swapchainExtent.height;
    pc.farDepth = 1.0f;

    vkCmdPushConstants(
        cmd,
        meshTilePipeline->getPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(MeshTilePC), &pc
    );

    vkCmdDispatch(cmd, meshTileExtent.width, meshTileExtent.height, 1);

    {
        VkImage img = meshTileDepth[imageIndex]->image;

        VkImageMemoryBarrier toRead{};
        toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image = img;
        toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toRead.subresourceRange.baseMipLevel = 0;
        toRead.subresourceRange.levelCount = 1;
        toRead.subresourceRange.baseArrayLayer = 0;
        toRead.subresourceRange.layerCount = 1;
        toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &toRead
        );

        meshTileDepthLayouts[imageIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "gs_mesh_tile_end");
    }
}

void GsMixRenderPass::dispatchPreprocess(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex)
{
    const uint32_t N = static_cast<uint32_t>(scene->getNumVertices());
    const uint32_t numGroups = (N + 255u) / 256u;

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_preprocess_start");
    }

    const bool useMix =
        (occluder != nullptr) &&
        (preprocessPipelineMix != nullptr) &&
        (preprocessMixDepthSet != nullptr) &&
        (!meshTileDepth.empty());

    uint8_t curFrame = static_cast<uint8_t>(frameIndex % FRAMES_IN_FLIGHT);

    if (useMix) {
        preprocessPipelineMix->bind(
            cmd,
            curFrame,
            Pipeline::DescriptorOption({ 0u, 0u, imageIndex })
        );
    }
    else {
        preprocessPipeline->bind(cmd, frameIndex, /*variant*/ 0);
    }

    vkCmdDispatch(cmd, numGroups, 1, 1);

    tileOverlapBuffer->computeWriteReadBarrier(cmd);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    const VkDeviceSize copySize =
        std::min(tileOverlapBuffer->size, prefixSumPingBuffer->size);
    copyRegion.size = copySize;

    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = tileOverlapBuffer->buffer;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        1, &b,
        0, nullptr
    );

    vkCmdCopyBuffer(
        cmd,
        tileOverlapBuffer->buffer,
        prefixSumPingBuffer->buffer,
        1,
        &copyRegion
    );

    prefixSumPingBuffer->computeWriteReadBarrier(cmd);

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_preprocess_end");
    }
}

void GsMixRenderPass::ensureSortCapacity(uint32_t requiredInstances)
{
    const uint32_t N = static_cast<uint32_t>(scene->getNumVertices());
    if (N == 0) return;

    const uint64_t capacity = static_cast<uint64_t>(N) * static_cast<uint64_t>(sortBufferSizeMultiplier);
    if (static_cast<uint64_t>(requiredInstances) <= capacity) {
        return;
    }

    const auto old = sortBufferSizeMultiplier;

    uint32_t needMul = (requiredInstances + N - 1u) / N;
    if (needMul < 1u) needMul = 1u;

    uint32_t newMul = sortBufferSizeMultiplier;
    while (newMul < needMul) {
        newMul = std::max(newMul * 2u, needMul);
    }

    sortBufferSizeMultiplier = newMul;

    spdlog::warn("[GsMixRenderPass] Reallocating sort buffers. multiplier {} -> {} (requiredInstances={}, capacity={} -> {})",
        old, sortBufferSizeMultiplier, requiredInstances,
        static_cast<unsigned long long>(capacity),
        static_cast<unsigned long long>(static_cast<uint64_t>(N) * sortBufferSizeMultiplier));

    const uint64_t sortElementCount =
        static_cast<uint64_t>(N) * static_cast<uint64_t>(sortBufferSizeMultiplier);

    sortKBufferEven->realloc(sortElementCount * sizeof(uint64_t));
    sortKBufferOdd->realloc(sortElementCount * sizeof(uint64_t));
    sortVBufferEven->realloc(sortElementCount * sizeof(uint32_t));
    sortVBufferOdd->realloc(sortElementCount * sizeof(uint32_t));

    uint32_t globalInvocationSize =
        static_cast<uint32_t>(sortElementCount / numRadixSortBlocksPerWorkgroup);

    const uint32_t remainder =
        static_cast<uint32_t>(sortElementCount % numRadixSortBlocksPerWorkgroup);

    if (remainder > 0) globalInvocationSize++;

    const uint32_t numWorkgroups = (globalInvocationSize + 256u - 1u) / 256u;

    sortHistBuffer->realloc(static_cast<uint64_t>(numWorkgroups) * 256ull * sizeof(uint32_t));

}

void GsMixRenderPass::dispatchPrefixSum(VkCommandBuffer cmd, uint32_t frameIndex)
{
    const uint32_t N = static_cast<uint32_t>(scene->getNumVertices());
    const uint32_t numGroups = (N + 255u) / 256u;

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex,
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_prefix_sum_start");
    }

    prefixSumPipeline->bind(cmd, frameIndex, 0);

    const uint32_t iters = static_cast<uint32_t>(
        std::ceil(std::log2(static_cast<float>(N)))
        );

    for (uint32_t timestep = 0; timestep <= iters; ++timestep) {
        vkCmdPushConstants(
            cmd,
            prefixSumPipeline->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(uint32_t),
            &timestep);

        vkCmdDispatch(cmd, numGroups, 1, 1);

        if ((timestep & 1u) == 0u) {
            prefixSumPongBuffer->computeWriteReadBarrier(cmd);
            prefixSumPingBuffer->computeReadWriteBarrier(cmd);
        }
        else {
            prefixSumPingBuffer->computeWriteReadBarrier(cmd);
            prefixSumPongBuffer->computeReadWriteBarrier(cmd);
        }
    }

    VkBuffer finalSrc =
        ((iters & 1u) == 0u) ? prefixSumPongBuffer->buffer : prefixSumPingBuffer->buffer;

    VkBufferMemoryBarrier b0{};
    b0.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b0.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b0.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.buffer = finalSrc;
    b0.offset = 0;
    b0.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        1, &b0,
        0, nullptr
    );

    VkBufferCopy totalSumRegion{};
    totalSumRegion.size = sizeof(uint32_t);
    totalSumRegion.srcOffset = static_cast<VkDeviceSize>(N - 1) * sizeof(uint32_t);
    totalSumRegion.dstOffset = 0;

    vkCmdCopyBuffer(
        cmd,
        finalSrc,
        totalSumBufferHost->buffer,
        1,
        &totalSumRegion
    );

    VkBufferMemoryBarrier b1{};
    b1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.buffer = totalSumBufferHost->buffer;
    b1.offset = 0;
    b1.size = sizeof(uint32_t);

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0, nullptr,
        1, &b1,
        0, nullptr
    );

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex,
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_prefix_sum_end");
    }
}

void GsMixRenderPass::dispatchPreprocessSort(VkCommandBuffer cmd, uint32_t frameIndex)
{
    const uint32_t N = static_cast<uint32_t>(scene->getNumVertices());
    const uint32_t numGroups = (N + 255u) / 256u;

    const uint32_t iters = static_cast<uint32_t>(
        std::ceil(std::log2(static_cast<float>(N)))
        );

    const uint32_t tileX = (swapchain->swapchainExtent.width + 16u - 1u) / 16u;

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_preprocess_sort_start");
    }

    preprocessSortPipeline->bind(cmd, frameIndex, (iters & 1u) ? 1u : 0u);

    vkCmdPushConstants(
        cmd,
        preprocessSortPipeline->getPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(uint32_t),
        &tileX);

    vkCmdDispatch(cmd, numGroups, 1, 1);

    sortKBufferEven->computeWriteReadBarrier(cmd);

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_preprocess_sort_end");
    }
}

void GsMixRenderPass::dispatchRadixSort(VkCommandBuffer cmd, uint32_t frameIndex)
{
    const uint32_t numInstances = cachedNumInstances;

    if (numInstances == 0) {
        return;
    }

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_sort_start");
    }

    for (uint32_t i = 0; i < 8; ++i) {
        uint32_t invocationSize =
            (numInstances + numRadixSortBlocksPerWorkgroup - 1u) / numRadixSortBlocksPerWorkgroup;
        invocationSize = (invocationSize + 255u) / 256u;

        RadixSortPushConstants pc{};
        pc.g_num_elements = numInstances;
        pc.g_num_blocks_per_workgroup = numRadixSortBlocksPerWorkgroup;
        pc.g_shift = i * 8u;
        pc.g_num_workgroups = invocationSize;

        sortHistPipeline->bind(cmd, frameIndex, (i & 1u) ? 1u : 0u);

        vkCmdPushConstants(
            cmd,
            sortHistPipeline->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(RadixSortPushConstants),
            &pc);

        vkCmdDispatch(cmd, invocationSize, 1, 1);

        sortHistBuffer->computeWriteReadBarrier(cmd);

        sortPipeline->bind(cmd, frameIndex, (i & 1u) ? 1u : 0u);

        vkCmdPushConstants(
            cmd,
            sortPipeline->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(RadixSortPushConstants),
            &pc);

        vkCmdDispatch(cmd, invocationSize, 1, 1);

        if ((i & 1u) == 0u) {
            sortKBufferOdd->computeWriteReadBarrier(cmd);
            sortVBufferOdd->computeWriteReadBarrier(cmd);
        }
        else {
            sortKBufferEven->computeWriteReadBarrier(cmd);
            sortVBufferEven->computeWriteReadBarrier(cmd);
        }
    }

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_sort_end");
    }
}

void GsMixRenderPass::dispatchTileBoundary(VkCommandBuffer cmd, uint32_t frameIndex)
{
    const uint32_t numInstances = cachedNumInstances;
    if (numInstances == 0) {
        vkCmdFillBuffer(cmd, tileBoundaryBuffer->buffer, 0, VK_WHOLE_SIZE, 0);
        return;
    }

    vkCmdFillBuffer(cmd, tileBoundaryBuffer->buffer, 0, VK_WHOLE_SIZE, 0);

    VkBufferMemoryBarrier bufBarrier{};
    bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.buffer = tileBoundaryBuffer->buffer;
    bufBarrier.offset = 0;
    bufBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &bufBarrier,
        0, nullptr);

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_tile_boundary_start");
    }

    tileBoundaryPipeline->bind(cmd, frameIndex, /*variant*/ 0);

    vkCmdPushConstants(
        cmd,
        tileBoundaryPipeline->getPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(uint32_t),
        &numInstances);

    vkCmdDispatch(cmd, (numInstances + 255u) / 256u, 1, 1);

    tileBoundaryBuffer->computeWriteReadBarrier(cmd);

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_tile_boundary_end");
    }
}

void GsMixRenderPass::dispatchRender(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex)
{
    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex,
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            "gs_render_start");
    }

    const bool useMix =
        (renderPipelineMix != nullptr) &&
        (renderDepthSet != nullptr) &&
        (occluder != nullptr);

    auto pipe = useMix ? renderPipelineMix : renderPipeline;

    uint8_t curFrame = static_cast<uint8_t>(frameIndex % FRAMES_IN_FLIGHT);

    if (useMix) {
        pipe->bind(
            cmd,
            curFrame,
            Pipeline::DescriptorOption({ 0u, imageIndex, imageIndex })
        );
    }
    else {
        pipe->bind(
            cmd,
            curFrame,
            Pipeline::DescriptorOption({ 0u, imageIndex })
        );
    }

    auto [width, height] = swapchain->swapchainExtent;

    if (useMix) {
        MixPC pc{};
        pc.width = width;
        pc.height = height;

        if (!camera) {
            throw std::runtime_error("GsMixRenderPass::dispatchRender: camera is null");
        }

        glm::mat4 P = ComputeProjGltf_FromCameraLikeRenderer(*camera, width, height);

        pc.projA = P[2][2];
        pc.projB = P[3][2];
        pc.projC = P[2][3];

        pc.farDepth = 1.0f;

        vkCmdPushConstants(
            cmd,
            renderPipelineMix->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(MixPC), &pc
        );
    }
    else {
        uint32_t pc[2] = { width, height };
        vkCmdPushConstants(
            cmd,
            renderPipeline->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), pc
        );
    }

    VkImage colorImage = gsColor[imageIndex]->image;

    {
        VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags        srcAccess = 0;
        InferSrcStageAccess(gsColorLayouts[imageIndex], srcStage, srcAccess);

        VkImageMemoryBarrier toGeneral{};
        toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toGeneral.oldLayout = gsColorLayouts[imageIndex];
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = colorImage;
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.baseMipLevel = 0;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.baseArrayLayer = 0;
        toGeneral.subresourceRange.layerCount = 1;

        toGeneral.srcAccessMask = srcAccess;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &toGeneral
        );

        gsColorLayouts[imageIndex] = VK_IMAGE_LAYOUT_GENERAL;
    }

    if (useMix) {
        if (gsDepth.empty() || !gsDepth[imageIndex]) {
            throw std::runtime_error("GsMixRenderPass::dispatchRender: useMix but gsDepth[imageIndex] is null");
        }

        VkImage depthImage = gsDepth[imageIndex]->image;

        VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags        srcAccess = 0;
        InferSrcStageAccess(gsDepthLayouts[imageIndex], srcStage, srcAccess);

        VkImageMemoryBarrier toGeneral{};
        toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toGeneral.oldLayout = gsDepthLayouts[imageIndex];
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = depthImage;
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.baseMipLevel = 0;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.baseArrayLayer = 0;
        toGeneral.subresourceRange.layerCount = 1;

        toGeneral.srcAccessMask = srcAccess;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            srcStage,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &toGeneral
        );

        gsDepthLayouts[imageIndex] = VK_IMAGE_LAYOUT_GENERAL;
    }

    vkCmdDispatch(
        cmd,
        (width + 15u) / 16u,
        (height + 15u) / 16u,
        1
    );

    {
        VkImageMemoryBarrier toRead{};
        toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image = colorImage;
        toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toRead.subresourceRange.baseMipLevel = 0;
        toRead.subresourceRange.levelCount = 1;
        toRead.subresourceRange.baseArrayLayer = 0;
        toRead.subresourceRange.layerCount = 1;

        toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &toRead
        );

        gsColorLayouts[imageIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    if (useMix) {
        VkImage depthImage = gsDepth[imageIndex]->image;

        VkImageMemoryBarrier toRead{};
        toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image = depthImage;
        toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toRead.subresourceRange.baseMipLevel = 0;
        toRead.subresourceRange.levelCount = 1;
        toRead.subresourceRange.baseArrayLayer = 0;
        toRead.subresourceRange.layerCount = 1;

        toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &toRead
        );
        gsDepthLayouts[imageIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(frameIndex, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "gs_render_end");
    }
}