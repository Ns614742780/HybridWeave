#include "PresentPass.h"
#include "DescriptorSet.h"
#include "IGBufferProvider.h"
#include "IDepthProvider.h"
#include "Image.h"
#include "Shader.h"

#include <stdexcept>
#include <spdlog/spdlog.h>
#include <iostream>

static std::shared_ptr<Image> CreateImage2D(
    const std::shared_ptr<VulkanContext>& context,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspectMask,
    const char* debugName)
{
    auto out = std::make_shared<Image>();
    out->format = format;
    out->extent = { width, height };

    // ---- VkImage ----
    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { width, height, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = 0;

    if (vmaCreateImage(context->allocator, &ci, &ai, &out->image, &out->allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error(std::string("CreateImage2D failed: ") + debugName);
    }

    // ---- VkImageView ----
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = out->image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = aspectMask;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;

    if (vkCreateImageView(context->device, &vi, nullptr, &out->imageView) != VK_SUCCESS) {
        vmaDestroyImage(context->allocator, out->image, out->allocation);
        out->image = VK_NULL_HANDLE;
        out->allocation = VK_NULL_HANDLE;
        throw std::runtime_error(std::string("CreateImageView failed: ") + debugName);
    }
    out->mip0View = out->imageView;
    return out;
}

static void DestroyImage(
    const std::shared_ptr<VulkanContext>& context,
    std::shared_ptr<Image>& img)
{
    if (!img) return;
    if (img->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, img->imageView, nullptr);
        img->imageView = VK_NULL_HANDLE;
    }
    img->mip0View = VK_NULL_HANDLE;

    if (img->image != VK_NULL_HANDLE && img->allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, img->image, img->allocation);
        img->image = VK_NULL_HANDLE;
        img->allocation = VK_NULL_HANDLE;
    }
    img.reset();
}

PresentPass::PresentPass(const RenderGlobalResources& r, IColorProvider* srcA_, IColorProvider* srcB_, Params p)
    : global(r)
    , context(r.context)
    , swapchain(r.swapchain)
    , srcA(srcA_)
    , srcB(srcB_)
    , params(p)
{
    if (!context)  throw std::runtime_error("PresentPass: context is null");
    if (!swapchain) throw std::runtime_error("PresentPass: swapchain is null");
    if (!srcA) throw std::runtime_error("PresentPass: srcA(IColorProvider) is null");

    if (params.mode == Mode::Mix) {
        if (!srcB) {
            spdlog::warn("PresentPass: mode requires srcB but srcB is null. Will fallback to srcA.");
        }
    }
}

PresentPass::~PresentPass()
{
    destroySwapchainDependentResources();
    destroyResources();

    destroyFullscreenTriangleVB();
    destroySampler();
    destroyRenderPassAndFramebuffers();
}

void PresentPass::initialize()
{
    spdlog::debug("PresentPass::initialize()");
    vkDeviceWaitIdle(context->device);

    createSampler();
    createFullscreenTriangleVB();
    createRenderPassAndFramebuffers();
    createPipelineAndDescriptors();
}

void PresentPass::onSwapchainResized()
{
    spdlog::debug("PresentPass::onSwapchainResized()");
    vkDeviceWaitIdle(context->device);

    destroyRenderPassAndFramebuffers();
    destroySwapchainDependentResources();

    createRenderPassAndFramebuffers();

    createBlurMapImages(FRAMES_IN_FLIGHT);
    rebuildPresentDescriptorsAndPipelineOnly();
}

void PresentPass::update(float) {}

void PresentPass::createSampler()
{
    if (srcSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod = 0.0f;
        si.maxLod = 0.0f;
        si.anisotropyEnable = VK_FALSE;
        if (vkCreateSampler(context->device, &si, nullptr, &srcSampler) != VK_SUCCESS) {
            throw std::runtime_error("PresentPass: failed to create sampler");
        }
    }
    if (nearestSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo ni{};
        ni.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ni.magFilter = VK_FILTER_NEAREST;
        ni.minFilter = VK_FILTER_NEAREST;
        ni.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ni.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ni.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ni.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ni.minLod = 0.0f;
        ni.maxLod = 0.0f;
        if (vkCreateSampler(context->device, &ni, nullptr, &nearestSampler) != VK_SUCCESS) {
            throw std::runtime_error("PresentPass: failed to create nearestSampler");
        }
    }
}

void PresentPass::destroySampler()
{
    if (srcSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, srcSampler, nullptr);
        srcSampler = VK_NULL_HANDLE;
    }
    if (nearestSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, nearestSampler, nullptr);
        nearestSampler = VK_NULL_HANDLE;
	}
}

void PresentPass::destroySwapchainDependentResources()
{
    // graphics pipeline + present sets
    pipeline.reset();
    srcSet.reset();
    enhanceSet.reset();

    // blurmap (full-res) + sets/pipeline
    blurMapPipeline.reset();
    blurMapSet.reset();
    blurMapFragSet.reset();

    for (uint32_t f = 0; f < blurMapPrev.size(); ++f) DestroyImage(context, blurMapPrev[f]);
    for (uint32_t f = 0; f < blurMapNext.size(); ++f) DestroyImage(context, blurMapNext[f]);
    blurMapPrev.clear();
    blurMapNext.clear();
    blurMapPrevLayouts.clear();
    blurMapNextLayouts.clear();
}

void PresentPass::destroyResources()
{
    autoMatchPipeline.reset();
    autoMatchSet.reset();

    downsamplePipeline.reset();
    downsampleSet.reset();

    for (auto& buf : autoMatchParams) buf.reset();
    autoMatchParams.clear();

    for (uint32_t f = 0; f < autoMatchLutPrev.size(); ++f) DestroyImage(context, autoMatchLutPrev[f]);
    for (uint32_t f = 0; f < autoMatchLutNext.size(); ++f) DestroyImage(context, autoMatchLutNext[f]);
    autoMatchLutPrev.clear();
    autoMatchLutNext.clear();
    autoMatchLutPrevLayouts.clear();
    autoMatchLutNextLayouts.clear();

    for (uint32_t f = 0; f < statsColorA.size(); ++f) DestroyImage(context, statsColorA[f]);
    for (uint32_t f = 0; f < statsColorB.size(); ++f) DestroyImage(context, statsColorB[f]);
    statsColorA.clear();
    statsColorB.clear();
    statsColorALayouts.clear();
    statsColorBLayouts.clear();
}

void PresentPass::createFullscreenTriangleVB()
{
    struct Vtx { float x, y, u, v; };
    const Vtx verts[3] = {
        { -1.0f, -1.0f, 0.0f, 0.0f },
        {  3.0f, -1.0f, 2.0f, 0.0f },
        { -1.0f,  3.0f, 0.0f, 2.0f },
    };

    fsTriangleVB = Buffer::vertex(
        context,
        sizeof(verts),
        /*concurrentSharing=*/false,
        "PresentPass Fullscreen Triangle VB"
    );

    fsTriangleVB->upload(verts, static_cast<uint32_t>(sizeof(verts)));
}

void PresentPass::destroyFullscreenTriangleVB()
{
    fsTriangleVB.reset();
}

void PresentPass::createRenderPassAndFramebuffers()
{
    const size_t imageCount = swapchain->swapchainImages.size();

    VkAttachmentDescription color{};
    color.format = swapchain->swapchainFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = 0;

    VkRenderPassCreateInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 1;
    rp.pAttachments = &color;
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 2;
    rp.pDependencies = deps;

    if (vkCreateRenderPass(context->device, &rp, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("PresentPass: failed to create render pass");
    }

    framebuffers.clear();
    framebuffers.resize(imageCount, VK_NULL_HANDLE);

    for (size_t i = 0; i < imageCount; ++i) {
        VkImageView views[1] = { swapchain->swapchainImages[i]->imageView };

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = renderPass;
        fb.attachmentCount = 1;
        fb.pAttachments = views;
        fb.width = swapchain->swapchainExtent.width;
        fb.height = swapchain->swapchainExtent.height;
        fb.layers = 1;

        if (vkCreateFramebuffer(context->device, &fb, nullptr, &framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("PresentPass: failed to create framebuffer");
        }
    }

    swapImageLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);
}

void PresentPass::destroyRenderPassAndFramebuffers()
{
    if (!context) return;

    for (auto fb : framebuffers) {
        if (fb != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(context->device, fb, nullptr);
        }
    }
    framebuffers.clear();

    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(context->device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }

    swapImageLayouts.clear();
}

void PresentPass::createStage2StatsImages(uint32_t frames)
{
    statsColorA.clear();
    statsColorB.clear();
    statsColorA.resize(frames);
    statsColorB.resize(frames);

    statsColorALayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);
    statsColorBLayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);

    for (uint32_t f = 0; f < frames; ++f) {
        statsColorA[f] = CreateImage2D(
            context,
            statsW, statsH,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "Stage2_StatsColorA"
        );

        statsColorB[f] = CreateImage2D(
            context,
            statsW, statsH,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "Stage2_StatsColorB"
        );
    }
}

void PresentPass::createStage2DownsamplePipeline(uint32_t frames)
{
    downsamplePipeline.reset();
    downsampleSet.reset();

    const uint32_t imageCount = static_cast<uint32_t>(swapchain->swapchainImages.size());

    downsampleSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);

    auto getColorSafe = [&](IColorProvider* p, uint32_t idx, const std::shared_ptr<Image>& fallback) -> std::shared_ptr<Image> {
        if (!p) return fallback;
        auto img = p->getColorImage(idx);
        return img ? img : fallback;
        };

    auto fallbackColorA = srcA ? srcA->getColorImage(0) : nullptr;
    if (!fallbackColorA) {
        throw std::runtime_error("PresentPass(Stage2): srcA->getColorImage(0) is null");
    }

    for (uint32_t f = 0; f < frames; ++f) {
        for (uint32_t i = 0; i < imageCount; ++i) {

            auto imgA = getColorSafe(srcA, i, fallbackColorA);
            auto imgB = imgA;
            if (srcB) imgB = getColorSafe(srcB, i, imgA);

            // sampled full-res inputs
            downsampleSet->bindCombinedImageSamplerToDescriptorSet(
                0, VK_SHADER_STAGE_COMPUTE_BIT,
                imgA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            downsampleSet->bindCombinedImageSamplerToDescriptorSet(
                1, VK_SHADER_STAGE_COMPUTE_BIT,
                imgB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            // storage outputs (per-frame stats targets)
            downsampleSet->bindStorageImageToDescriptorSet(
                2, VK_SHADER_STAGE_COMPUTE_BIT,
                statsColorA[f], VK_IMAGE_LAYOUT_GENERAL);

            downsampleSet->bindStorageImageToDescriptorSet(
                3, VK_SHADER_STAGE_COMPUTE_BIT,
                statsColorB[f], VK_IMAGE_LAYOUT_GENERAL);
        }
    }

    downsampleSet->build();

    downsamplePipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "stats_downsample")
    );
    downsamplePipeline->addDescriptorSet(0, downsampleSet);
    downsamplePipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DownsamplePC));
    downsamplePipeline->build();
}

void PresentPass::createBlurMapImages(uint32_t frames)
{
    // Destroy old if any
    for (uint32_t f = 0; f < blurMapPrev.size(); ++f) {
        DestroyImage(context, blurMapPrev[f]);
    }
    for (uint32_t f = 0; f < blurMapNext.size(); ++f) {
        DestroyImage(context, blurMapNext[f]);
    }

    blurMapPrev.clear();
    blurMapNext.clear();
    blurMapPrevLayouts.clear();
    blurMapNextLayouts.clear();

    blurMapPrev.resize(frames);
    blurMapNext.resize(frames);

    blurMapPrevLayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);
    blurMapNextLayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);

    const uint32_t W = swapchain->swapchainExtent.width;
    const uint32_t H = swapchain->swapchainExtent.height;

    for (uint32_t f = 0; f < frames; ++f) {
        blurMapPrev[f] = CreateImage2D(
            context,
            W, H,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "BlurMap_Prev"
        );

        blurMapNext[f] = CreateImage2D(
            context,
            W, H,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "BlurMap_Next"
        );
    }
}

void PresentPass::rebuildPresentDescriptorsAndPipelineOnly()
{
    vkDeviceWaitIdle(context->device);

    const uint32_t imageCount = static_cast<uint32_t>(swapchain->swapchainImages.size());
    const uint32_t frames = FRAMES_IN_FLIGHT;


    if (!srcA) throw std::runtime_error("PresentPass::rebuildPresentDescriptorsAndPipelineOnly: srcA is null");
    auto fallbackColorA = srcA->getColorImage(0);
    if (!fallbackColorA) throw std::runtime_error("PresentPass::rebuildPresentDescriptorsAndPipelineOnly: srcA->getColorImage(0) is null");

    gsDepthProvider = nullptr;
    if (params.mode == Mode::Mix) {
        if (!srcB) throw std::runtime_error("PresentPass: Mode::Mix requires srcB but it's null");
        gsDepthProvider = dynamic_cast<IDepthProvider*>(srcB);
        if (!gsDepthProvider) throw std::runtime_error("PresentPass: Mode::Mix requires srcB to implement IDepthProvider");
        if (!gbufferProvider) throw std::runtime_error("PresentPass: Mode::Mix requires gbufferProvider but it's null");
    }

    srcSet.reset();
    enhanceSet.reset();
    pipeline.reset();

    blurMapSet.reset();
    blurMapFragSet.reset();

    srcSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);
    enhanceSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);
    blurMapSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);
    blurMapFragSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);

    auto getColorSafe = [&](IColorProvider* p, uint32_t idx, const std::shared_ptr<Image>& fallback) -> std::shared_ptr<Image> {
        if (!p) return fallback;
        auto img = p->getColorImage(idx);
        return img ? img : fallback;
        };

    for (uint32_t f = 0; f < frames; ++f) {
        for (uint32_t i = 0; i < imageCount; ++i) {
            for (uint32_t ping = 0; ping < 2; ++ping) {

                auto imgA = getColorSafe(srcA, i, fallbackColorA);
                auto imgB = imgA;
                if (srcB) imgB = getColorSafe(srcB, i, imgA);

                // Depth A/B
                std::shared_ptr<Image> depthA = imgA;
                std::shared_ptr<Image> depthB = imgA;

                if (params.mode == Mode::Mix) {
                    const auto& gbufs = gbufferProvider->getGBufferImages();
                    if (i >= gbufs.size() || !gbufs[i].depth) {
                        throw std::runtime_error("PresentPass: gbuffer depth missing or size mismatch");
                    }
                    depthA = gbufs[i].depth;

                    auto gsD = gsDepthProvider->getDepthImage(i);
                    if (!gsD) throw std::runtime_error("PresentPass: 3DGS depth image is null");
                    depthB = gsD;
                }

                auto lutPrev = (ping == 0) ? autoMatchLutPrev[f] : autoMatchLutNext[f];
                auto lutNext = (ping == 0) ? autoMatchLutNext[f] : autoMatchLutPrev[f];

                // BlurMap ping-pong (swapchain dependent)
                auto bmPrev = (ping == 0) ? blurMapPrev[f] : blurMapNext[f];
                auto bmNext = (ping == 0) ? blurMapNext[f] : blurMapPrev[f];

                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    0, VK_SHADER_STAGE_FRAGMENT_BIT,
                    imgA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    1, VK_SHADER_STAGE_FRAGMENT_BIT,
                    imgB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    2, VK_SHADER_STAGE_FRAGMENT_BIT,
                    depthA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    3, VK_SHADER_STAGE_FRAGMENT_BIT,
                    depthB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                enhanceSet->bindBufferToDescriptorSet(
                    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT, autoMatchParams[f]);

                enhanceSet->bindCombinedImageSamplerToDescriptorSet(
                    1, VK_SHADER_STAGE_FRAGMENT_BIT,
                    lutNext, nearestSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    0, VK_SHADER_STAGE_COMPUTE_BIT,
                    imgA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    1, VK_SHADER_STAGE_COMPUTE_BIT,
                    depthA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    2, VK_SHADER_STAGE_COMPUTE_BIT,
                    imgB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    3, VK_SHADER_STAGE_COMPUTE_BIT,
                    depthB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    4, VK_SHADER_STAGE_COMPUTE_BIT,
                    bmPrev, nearestSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                blurMapSet->bindStorageImageToDescriptorSet(
                    5, VK_SHADER_STAGE_COMPUTE_BIT,
                    bmNext, VK_IMAGE_LAYOUT_GENERAL);

                blurMapFragSet->bindCombinedImageSamplerToDescriptorSet(
                    0, VK_SHADER_STAGE_FRAGMENT_BIT,
                    bmNext, nearestSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }

    srcSet->build();
    enhanceSet->build();
    blurMapSet->build();
    blurMapFragSet->build();

    if (!blurMapPipeline) {
        blurMapPipeline = std::make_shared<ComputePipeline>(
            context,
            std::make_shared<Shader>(context, "blurmap_build")
        );
        blurMapPipeline->addDescriptorSet(0, blurMapSet);
        blurMapPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurMapPC));
        blurMapPipeline->build();
    }
    else {

        blurMapPipeline.reset();
        blurMapPipeline = std::make_shared<ComputePipeline>(
            context,
            std::make_shared<Shader>(context, "blurmap_build")
        );
        blurMapPipeline->addDescriptorSet(0, blurMapSet);
        blurMapPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurMapPC));
        blurMapPipeline->build();
    }

    pipeline = std::make_shared<GraphicsPipeline>(context);
    pipeline->setDepthTest(false);
    pipeline->addDescriptorSet(0, srcSet);

    if (params.openMixEnhance) {
        pipeline->addDescriptorSet(1, enhanceSet);
        if (params.enableBlurMap) {
            pipeline->addDescriptorSet(2, blurMapFragSet);
        }
    }

    pipeline->addPushConstant(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConst));
    pipeline->setRenderPass(renderPass, 0);
    pipeline->setExtent(swapchain->swapchainExtent);

    GraphicsPipeline::VertexInputDesc vi{};
    vi.binding.binding = 0;
    vi.binding.stride = sizeof(float) * 4;
    vi.binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    vi.attributes.resize(2);
    vi.attributes[0].location = 0;
    vi.attributes[0].binding = 0;
    vi.attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    vi.attributes[0].offset = 0;

    vi.attributes[1].location = 1;
    vi.attributes[1].binding = 0;
    vi.attributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    vi.attributes[1].offset = sizeof(float) * 2;

    pipeline->setVertexInput(vi);

    auto vert = std::make_shared<Shader>(context, "lighting_vert");

    const bool useMixAlpha = (params.mode == Mode::Mix);
    std::string fragName;
    if (useMixAlpha) fragName = params.openMixEnhance ? "present_mix_enhance" : "present_mix";
    else fragName = "present_frag";

    auto frag = std::make_shared<Shader>(context, fragName);
    pipeline->setShaders(vert, frag);

    pipeline->setCullMode(VK_CULL_MODE_NONE);
    pipeline->setFrontFace(VK_FRONT_FACE_CLOCKWISE);

    std::vector<VkPipelineColorBlendAttachmentState> blends(1);
    blends[0].blendEnable = VK_FALSE;
    blends[0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    pipeline->setColorBlendAttachments(blends);

    pipeline->build();
}

void PresentPass::createPipelineAndDescriptors()
{
    vkDeviceWaitIdle(context->device);

    const uint32_t imageCount = static_cast<uint32_t>(swapchain->swapchainImages.size());
    const uint32_t frames = FRAMES_IN_FLIGHT;


    autoMatchPing.assign(frames, 0u);

    autoMatchParams.clear();
    autoMatchParams.resize(frames);

    for (uint32_t f = 0; f < frames; ++f) {
        autoMatchParams[f] = Buffer::storage(
            context,
            sizeof(AutoMatchParamsGPU),
            /*concurrentSharing=*/false,
            /*memoryFlags=*/0,
            "AutoMatchParams"
        );
        AutoMatchParamsGPU init{};
        init.gain = 1.0f;
        init.wbR = 1.0f;
        init.wbG = 1.0f;
        init.wbB = 1.0f;
        init.blurStats = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        autoMatchParams[f]->upload(&init, sizeof(AutoMatchParamsGPU));
    }

    autoMatchLutPrev.clear();
    autoMatchLutNext.clear();
    autoMatchLutPrev.resize(frames);
    autoMatchLutNext.resize(frames);

    autoMatchLutPrevLayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);
    autoMatchLutNextLayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);

    for (uint32_t f = 0; f < frames; ++f) {
        autoMatchLutPrev[f] = CreateImage2D(
            context,
            256, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "AutoMatchLUT_Prev"
        );

        autoMatchLutNext[f] = CreateImage2D(
            context,
            256, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "AutoMatchLUT_Next"
        );
    }

    createStage2StatsImages(frames);

    blurMapPrev.clear();
    blurMapNext.clear();
    blurMapPrev.resize(frames);
    blurMapNext.resize(frames);

    blurMapPrevLayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);
    blurMapNextLayouts.assign(frames, VK_IMAGE_LAYOUT_UNDEFINED);

    for (uint32_t f = 0; f < frames; ++f) {
        blurMapPrev[f] = CreateImage2D(
            context,
            swapchain->swapchainExtent.width,
            swapchain->swapchainExtent.height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "BlurMap_Prev"
        );

        blurMapNext[f] = CreateImage2D(
            context,
            swapchain->swapchainExtent.width,
            swapchain->swapchainExtent.height,
            VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            "BlurMap_Next"
        );
    }

    lutInitPipeline.reset();
    lutInitSet.reset();

    lutInitSet = std::make_shared<DescriptorSet>(context, 1);
    for (uint32_t f = 0; f < frames; ++f)
    {
        lutInitSet->bindStorageImageToDescriptorSet(
            0, VK_SHADER_STAGE_COMPUTE_BIT,
            autoMatchLutPrev[f], VK_IMAGE_LAYOUT_GENERAL
        );
        lutInitSet->bindStorageImageToDescriptorSet(
            0, VK_SHADER_STAGE_COMPUTE_BIT,
            autoMatchLutNext[f], VK_IMAGE_LAYOUT_GENERAL
        );
    }
    lutInitSet->build();

    lutInitPipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "lut_init")
    );
    lutInitPipeline->addDescriptorSet(0, lutInitSet);
    lutInitPipeline->build();

    gsDepthProvider = nullptr;
    if (params.mode == Mode::Mix) {
        if (!srcB) throw std::runtime_error("PresentPass: Mode::Mix requires srcB but it's null");
        gsDepthProvider = dynamic_cast<IDepthProvider*>(srcB);
        if (!gsDepthProvider) throw std::runtime_error("PresentPass: Mode::Mix requires srcB to implement IDepthProvider");
        if (!gbufferProvider) throw std::runtime_error("PresentPass: Mode::Mix requires gbufferProvider but it's null");
    }


    srcSet.reset();
    autoMatchSet.reset();
    enhanceSet.reset();
    pipeline.reset();
    autoMatchPipeline.reset();

    blurMapPipeline.reset();
    blurMapSet.reset();
    blurMapFragSet.reset();

    downsamplePipeline.reset();
    downsampleSet.reset();

    createStage2DownsamplePipeline(frames);

    srcSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);
    autoMatchSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);
    enhanceSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);
    blurMapSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);
    blurMapFragSet = std::make_shared<DescriptorSet>(context, (uint8_t)frames);

    // helper: safe fetch
    auto getColorSafe = [&](IColorProvider* p, uint32_t idx, const std::shared_ptr<Image>& fallback) -> std::shared_ptr<Image> {
        if (!p) return fallback;
        auto img = p->getColorImage(idx);
        return img ? img : fallback;
        };

    auto fallbackColorA = srcA ? srcA->getColorImage(0) : nullptr;
    if (!fallbackColorA) throw std::runtime_error("PresentPass: srcA->getColorImage(0) is null");

    for (uint32_t f = 0; f < frames; ++f) {
        for (uint32_t i = 0; i < imageCount; ++i) {
            for (uint32_t ping = 0; ping < 2; ++ping) {

                auto imgA = getColorSafe(srcA, i, fallbackColorA);
                auto imgB = imgA;
                if (srcB) imgB = getColorSafe(srcB, i, imgA);

                // Depth A/B
                std::shared_ptr<Image> depthA = imgA;
                std::shared_ptr<Image> depthB = imgA;

                if (params.mode == Mode::Mix) {
                    const auto& gbufs = gbufferProvider->getGBufferImages();
                    if (i >= gbufs.size() || !gbufs[i].depth) {
                        throw std::runtime_error("PresentPass: gbuffer depth missing or size mismatch");
                    }
                    depthA = gbufs[i].depth;

                    auto gsD = gsDepthProvider->getDepthImage(i);
                    if (!gsD) throw std::runtime_error("PresentPass: 3DGS depth image is null");
                    depthB = gsD;
                }

                // LUT ping-pong
                auto lutPrev = (ping == 0) ? autoMatchLutPrev[f] : autoMatchLutNext[f];
                auto lutNext = (ping == 0) ? autoMatchLutNext[f] : autoMatchLutPrev[f];

                // BlurMap ping-pong
                auto bmPrev = (ping == 0) ? blurMapPrev[f] : blurMapNext[f];
                auto bmNext = (ping == 0) ? blurMapNext[f] : blurMapPrev[f];


                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    0, VK_SHADER_STAGE_FRAGMENT_BIT,
                    imgA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    1, VK_SHADER_STAGE_FRAGMENT_BIT,
                    imgB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    2, VK_SHADER_STAGE_FRAGMENT_BIT,
                    depthA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                srcSet->bindCombinedImageSamplerToDescriptorSet(
                    3, VK_SHADER_STAGE_FRAGMENT_BIT,
                    depthB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


                autoMatchSet->bindCombinedImageSamplerToDescriptorSet(
                    0, VK_SHADER_STAGE_COMPUTE_BIT,
                    statsColorA[f], srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                autoMatchSet->bindCombinedImageSamplerToDescriptorSet(
                    1, VK_SHADER_STAGE_COMPUTE_BIT,
                    depthA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                autoMatchSet->bindCombinedImageSamplerToDescriptorSet(
                    2, VK_SHADER_STAGE_COMPUTE_BIT,
                    statsColorB[f], srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                autoMatchSet->bindCombinedImageSamplerToDescriptorSet(
                    3, VK_SHADER_STAGE_COMPUTE_BIT,
                    depthB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                autoMatchSet->bindBufferToDescriptorSet(
                    4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_SHADER_STAGE_COMPUTE_BIT, autoMatchParams[f]);

                autoMatchSet->bindCombinedImageSamplerToDescriptorSet(
                    5, VK_SHADER_STAGE_COMPUTE_BIT,
                    lutPrev, nearestSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                autoMatchSet->bindStorageImageToDescriptorSet(
                    6, VK_SHADER_STAGE_COMPUTE_BIT,
                    lutNext, VK_IMAGE_LAYOUT_GENERAL);

                enhanceSet->bindBufferToDescriptorSet(
                    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT, autoMatchParams[f]);
                enhanceSet->bindCombinedImageSamplerToDescriptorSet(
                    1, VK_SHADER_STAGE_FRAGMENT_BIT,
                    lutNext, nearestSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                // BlurMap uses full-res inputs (unchanged)
                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    0, VK_SHADER_STAGE_COMPUTE_BIT,
                    imgA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    1, VK_SHADER_STAGE_COMPUTE_BIT,
                    depthA, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    2, VK_SHADER_STAGE_COMPUTE_BIT,
                    imgB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    3, VK_SHADER_STAGE_COMPUTE_BIT,
                    depthB, srcSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                blurMapSet->bindCombinedImageSamplerToDescriptorSet(
                    4, VK_SHADER_STAGE_COMPUTE_BIT,
                    bmPrev, nearestSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                blurMapSet->bindStorageImageToDescriptorSet(
                    5, VK_SHADER_STAGE_COMPUTE_BIT,
                    bmNext, VK_IMAGE_LAYOUT_GENERAL);

                blurMapFragSet->bindCombinedImageSamplerToDescriptorSet(
                    0, VK_SHADER_STAGE_FRAGMENT_BIT,
                    bmNext, nearestSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }

    srcSet->build();
    autoMatchSet->build();
    enhanceSet->build();
    blurMapSet->build();
    blurMapFragSet->build();


    {
        autoMatchPipeline = std::make_shared<ComputePipeline>(
            context,
            std::make_shared<Shader>(context, "automatch_ring") // you will update to PC2
        );
        autoMatchPipeline->addDescriptorSet(0, autoMatchSet);
        autoMatchPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AutoMatchPC2));
        autoMatchPipeline->build();
    }

    // BlurMap compute pipeline (unchanged)
    {
        blurMapPipeline = std::make_shared<ComputePipeline>(
            context,
            std::make_shared<Shader>(context, "blurmap_build")
        );
        blurMapPipeline->addDescriptorSet(0, blurMapSet);
        blurMapPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurMapPC));
        blurMapPipeline->build();
    }

    pipeline = std::make_shared<GraphicsPipeline>(context);
    pipeline->setDepthTest(false);
    pipeline->addDescriptorSet(0, srcSet);

    if (params.openMixEnhance) {
        pipeline->addDescriptorSet(1, enhanceSet);
        if (params.enableBlurMap) {
            pipeline->addDescriptorSet(2, blurMapFragSet);
        }
    }

    pipeline->addPushConstant(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConst));
    pipeline->setRenderPass(renderPass, 0);
    pipeline->setExtent(swapchain->swapchainExtent);

    GraphicsPipeline::VertexInputDesc vi{};
    vi.binding.binding = 0;
    vi.binding.stride = sizeof(float) * 4;
    vi.binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    vi.attributes.resize(2);
    vi.attributes[0].location = 0;
    vi.attributes[0].binding = 0;
    vi.attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    vi.attributes[0].offset = 0;

    vi.attributes[1].location = 1;
    vi.attributes[1].binding = 0;
    vi.attributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    vi.attributes[1].offset = sizeof(float) * 2;

    pipeline->setVertexInput(vi);

    auto vert = std::make_shared<Shader>(context, "lighting_vert");

    const bool useMixAlpha = (params.mode == Mode::Mix);
    std::string fragName;
    if (useMixAlpha) fragName = params.openMixEnhance ? "present_mix_enhance" : "present_mix";
    else fragName = "present_frag";

    auto frag = std::make_shared<Shader>(context, fragName);
    pipeline->setShaders(vert, frag);

    pipeline->setCullMode(VK_CULL_MODE_NONE);
    pipeline->setFrontFace(VK_FRONT_FACE_CLOCKWISE);

    std::vector<VkPipelineColorBlendAttachmentState> blends(1);
    blends[0].blendEnable = VK_FALSE;
    blends[0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    pipeline->setColorBlendAttachments(blends);

    pipeline->build();
}

void PresentPass::destroyPipelineAndDescriptors()
{
    srcSet.reset();
    pipeline.reset();

    lutInitPipeline.reset();
    lutInitSet.reset();

    autoMatchPipeline.reset();
    autoMatchSet.reset();
    enhanceSet.reset();

    blurMapPipeline.reset();
    blurMapSet.reset();
    blurMapFragSet.reset();

    downsamplePipeline.reset();
    downsampleSet.reset();

    for (auto& buf : autoMatchParams) buf.reset();
    autoMatchParams.clear();

    for (uint32_t f = 0; f < autoMatchLutPrev.size(); ++f) {
        DestroyImage(context, autoMatchLutPrev[f]);
    }
    autoMatchLutPrev.clear();

    for (uint32_t f = 0; f < autoMatchLutNext.size(); ++f) {
        DestroyImage(context, autoMatchLutNext[f]);
    }
    autoMatchLutNext.clear();

    autoMatchLutPrevLayouts.clear();
    autoMatchLutNextLayouts.clear();

    for (uint32_t f = 0; f < blurMapPrev.size(); ++f) {
        DestroyImage(context, blurMapPrev[f]);
    }
    blurMapPrev.clear();

    for (uint32_t f = 0; f < blurMapNext.size(); ++f) {
        DestroyImage(context, blurMapNext[f]);
    }
    blurMapNext.clear();

    blurMapPrevLayouts.clear();
    blurMapNextLayouts.clear();

    // ---- Stage2 stats ----
    for (uint32_t f = 0; f < statsColorA.size(); ++f) {
        DestroyImage(context, statsColorA[f]);
    }
    statsColorA.clear();

    for (uint32_t f = 0; f < statsColorB.size(); ++f) {
        DestroyImage(context, statsColorB[f]);
    }
    statsColorB.clear();

    statsColorALayouts.clear();
    statsColorBLayouts.clear();
}

void PresentPass::record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex)
{
    auto imgA = srcA ? srcA->getColorImage(imageIndex) : nullptr;
    if (!imgA) return;

    const uint32_t imageCount = static_cast<uint32_t>(swapchain->swapchainImages.size());
    const uint32_t frames = FRAMES_IN_FLIGHT;
    const uint32_t safeFrame = frameIndex % frames;

    // ping for this safeFrame
    const uint32_t ping = (autoMatchPing.empty() ? 0u : (autoMatchPing[safeFrame] & 1u));
    const uint32_t opt = ((safeFrame * imageCount + imageIndex) * 2u + ping);

    // layout sanity (do not force transition here)
    if (srcA && srcA->getColorLayout(imageIndex) != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        spdlog::warn("PresentPass: srcA layout is not SHADER_READ_ONLY_OPTIMAL (got={})",
            int(srcA->getColorLayout(imageIndex)));
    }
    if (srcB && srcB->getColorLayout(imageIndex) != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        spdlog::warn("PresentPass: srcB layout is not SHADER_READ_ONLY_OPTIMAL (got={})",
            int(srcB->getColorLayout(imageIndex)));
    }

    if (!lutInited && params.mode == Mode::Mix && params.openMixEnhance)
    {
        // init all LUTs once
        // layout transition: all prev/next -> GENERAL
        for (uint32_t f = 0; f < FRAMES_IN_FLIGHT; ++f)
        {
            auto forceToGeneral = [&](std::shared_ptr<Image> img, VkImageLayout& tracked)
                {
                    if (!img) return;
                    VkImageMemoryBarrier ib{};
                    ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    ib.oldLayout = tracked;
                    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    ib.image = img->image;
                    ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    ib.subresourceRange.baseMipLevel = 0;
                    ib.subresourceRange.levelCount = 1;
                    ib.subresourceRange.baseArrayLayer = 0;
                    ib.subresourceRange.layerCount = 1;
                    ib.srcAccessMask = 0;
                    ib.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

                    vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &ib
                    );
                    tracked = VK_IMAGE_LAYOUT_GENERAL;
                };

            forceToGeneral(autoMatchLutPrev[f], autoMatchLutPrevLayouts[f]);
            forceToGeneral(autoMatchLutNext[f], autoMatchLutNextLayouts[f]);
        }

        // dispatch init for each LUT
        for (uint32_t f = 0; f < FRAMES_IN_FLIGHT; ++f)
        {
            // option indices: f*2+0 prev, f*2+1 next
            for (uint32_t which = 0; which < 2; ++which)
            {
                uint32_t optInit = f * 2 + which;
                lutInitPipeline->bindWithSet(
                    cmd,
                    0,       // frameIndex dummy
                    optInit,  // option
                    lutInitSet
                );
                // 256 threads => 4 workgroups of 64
                vkCmdDispatch(cmd, 4, 1, 1);
            }
        }

        // barrier: init writes -> subsequent compute/frag reads
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            1, &mb,
            0, nullptr,
            0, nullptr
        );

        for (uint32_t f = 0; f < FRAMES_IN_FLIGHT; ++f)
        {
            auto toReadOnly = [&](std::shared_ptr<Image> img, VkImageLayout& tracked)
                {
                    if (!img) return;
                    VkImageMemoryBarrier ib{};
                    ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    ib.oldLayout = tracked;
                    ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    ib.image = img->image;
                    ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    ib.subresourceRange.baseMipLevel = 0;
                    ib.subresourceRange.levelCount = 1;
                    ib.subresourceRange.baseArrayLayer = 0;
                    ib.subresourceRange.layerCount = 1;
                    ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                    vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &ib
                    );
                    tracked = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                };
            toReadOnly(autoMatchLutPrev[f], autoMatchLutPrevLayouts[f]);
            toReadOnly(autoMatchLutNext[f], autoMatchLutNextLayouts[f]);
        }

        lutInited = true;
    }

    VkImage swapImage = swapchain->swapchainImages[imageIndex]->image;
    {
        VkImageLayout oldLayout = swapImageLayouts[imageIndex];

        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = oldLayout;
        b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = swapImage;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.baseMipLevel = 0;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.baseArrayLayer = 0;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &b
        );

        swapImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // AutoMatch + BlurMap compute dispatch
    auto EnsureImageLayout = [&](std::shared_ptr<Image> img, VkImageLayout& trackedLayout,
        VkImageLayout newLayout, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
        VkAccessFlags srcAccess, VkAccessFlags dstAccess)
        {
            if (!img) return;
            if (trackedLayout == newLayout) return;

            VkImageMemoryBarrier ib{};
            ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            ib.oldLayout = trackedLayout;
            ib.newLayout = newLayout;
            ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.image = img->image;
            ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ib.subresourceRange.baseMipLevel = 0;
            ib.subresourceRange.levelCount = 1;
            ib.subresourceRange.baseArrayLayer = 0;
            ib.subresourceRange.layerCount = 1;
            ib.srcAccessMask = srcAccess;
            ib.dstAccessMask = dstAccess;

            vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &ib);
            trackedLayout = newLayout;
        };

    if (params.mode == Mode::Mix && params.openMixEnhance)
    {
        {
            const uint32_t optDS = safeFrame * imageCount + imageIndex;

            EnsureImageLayout(
                statsColorA[safeFrame], statsColorALayouts[safeFrame],
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                VK_ACCESS_SHADER_WRITE_BIT
            );

            EnsureImageLayout(
                statsColorB[safeFrame], statsColorBLayouts[safeFrame],
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                VK_ACCESS_SHADER_WRITE_BIT
            );

            // ---------------- FIX #1 ----------------
            // optDS already encodes safeFrame. Do NOT pass safeFrame again as frameIndex.
            downsamplePipeline->bindWithSet(
                cmd,
                0,     // frameIndex fixed
                optDS,  // option
                downsampleSet
            );

            DownsamplePC dpc{};
            dpc.fullW = (int)swapchain->swapchainExtent.width;
            dpc.fullH = (int)swapchain->swapchainExtent.height;
            dpc.statsW = (int)statsW;
            dpc.statsH = (int)statsH;

            vkCmdPushConstants(
                cmd,
                downsamplePipeline->getPipelineLayout(),
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                sizeof(DownsamplePC),
                &dpc
            );

            uint32_t gx = (statsW + 7) / 8;
            uint32_t gy = (statsH + 7) / 8;
            vkCmdDispatch(cmd, gx, gy, 1);

            // barrier: downsample write -> automatch read
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &mb,
                0, nullptr,
                0, nullptr
            );

            EnsureImageLayout(
                statsColorA[safeFrame], statsColorALayouts[safeFrame],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT
            );

            EnsureImageLayout(
                statsColorB[safeFrame], statsColorBLayouts[safeFrame],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT
            );
        }

        auto lutPrev = (ping == 0) ? autoMatchLutPrev[safeFrame] : autoMatchLutNext[safeFrame];
        auto lutNext = (ping == 0) ? autoMatchLutNext[safeFrame] : autoMatchLutPrev[safeFrame];

        VkImageLayout& lutPrevLayout = (ping == 0) ? autoMatchLutPrevLayouts[safeFrame] : autoMatchLutNextLayouts[safeFrame];
        VkImageLayout& lutNextLayout = (ping == 0) ? autoMatchLutNextLayouts[safeFrame] : autoMatchLutPrevLayouts[safeFrame];

        EnsureImageLayout(
            lutPrev, lutPrevLayout,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            VK_ACCESS_SHADER_READ_BIT
        );

        // Next LUT should be writable
        EnsureImageLayout(
            lutNext, lutNextLayout,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            VK_ACCESS_SHADER_WRITE_BIT
        );

        // ---------------- FIX #1 ----------------
        // opt already encodes safeFrame. Do NOT pass safeFrame again as frameIndex.
        autoMatchPipeline->bindWithSet(
            cmd,
            0,   // frameIndex fixed
            opt, // option
            autoMatchSet
        );

        AutoMatchPC2 apc{};
        apc.statsW = (int)statsW;
        apc.statsH = (int)statsH;
        apc.fullW = (int)swapchain->swapchainExtent.width;
        apc.fullH = (int)swapchain->swapchainExtent.height;

        apc.ringRadius = 10;   // stats res smaller -> smaller radius
        apc.ringSamples = 24;
        apc.stride = 2;        // stable sampling in stats domain
        apc.styleLock = params.styleLock;

        apc.coverageTh = 0.65f;
        apc.emaAlpha = 0.18f;
        apc.maxDeltaGain = 0.05f;
        apc.maxDeltaWb = 0.05f;

        apc.minGain = 0.85f;
        apc.maxGain = 1.15f;

        vkCmdPushConstants(
            cmd,
            autoMatchPipeline->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(AutoMatchPC2),
            &apc
        );

        vkCmdDispatch(cmd, 1, 1, 1);

        // barrier: compute writes -> fragment reads (for lutNext)
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkImageMemoryBarrier ibLut{};
        ibLut.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ibLut.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        ibLut.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ibLut.image = lutNext->image;
        ibLut.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ibLut.subresourceRange.baseMipLevel = 0;
        ibLut.subresourceRange.levelCount = 1;
        ibLut.subresourceRange.baseArrayLayer = 0;
        ibLut.subresourceRange.layerCount = 1;
        ibLut.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ibLut.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            1, &mb,
            0, nullptr,
            1, &ibLut
        );
        lutNextLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // ---- BlurMap compute ----
        if (params.enableBlurMap)
        {
            auto bmPrev = (ping == 0) ? blurMapPrev[safeFrame] : blurMapNext[safeFrame];
            auto bmNext = (ping == 0) ? blurMapNext[safeFrame] : blurMapPrev[safeFrame];

            VkImageLayout& bmPrevLayout = (ping == 0) ? blurMapPrevLayouts[safeFrame] : blurMapNextLayouts[safeFrame];
            VkImageLayout& bmNextLayout = (ping == 0) ? blurMapNextLayouts[safeFrame] : blurMapPrevLayouts[safeFrame];

            EnsureImageLayout(
                bmPrev, bmPrevLayout,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                VK_ACCESS_SHADER_READ_BIT
            );

            EnsureImageLayout(
                bmNext, bmNextLayout,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                VK_ACCESS_SHADER_WRITE_BIT
            );

            // ---------------- FIX #1 ----------------
            blurMapPipeline->bindWithSet(
                cmd,
                0,   // frameIndex fixed
                opt, // option
                blurMapSet
            );

            BlurMapPC bpc{};
            bpc.width = (int)swapchain->swapchainExtent.width;
            bpc.height = (int)swapchain->swapchainExtent.height;

            // ---- stable defaults (tunable) ----
            bpc.emaAlpha = 0.08f;      // stable, low flicker
            bpc.covTh = 0.15f;         // low coverage threshold
            bpc.covStrongTh = 0.65f;
            bpc.baseSoft = 0.12f;      // IMPORTANT: ensures visible effect even if sparse
            bpc.maxSoft = 0.85f;

            bpc.depthTolBase = 1.5e-3f;
            bpc.depthTolK = 1.2e-2f;

            bpc.styleLock = params.styleLock;
            bpc.preset = params.stylePreset;

            vkCmdPushConstants(
                cmd,
                blurMapPipeline->getPipelineLayout(),
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                sizeof(BlurMapPC),
                &bpc
            );

            uint32_t gx = (swapchain->swapchainExtent.width + 7) / 8;
            uint32_t gy = (swapchain->swapchainExtent.height + 7) / 8;
            vkCmdDispatch(cmd, gx, gy, 1);

            // barrier + transition for bmNext: GENERAL -> READ_ONLY
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            VkImageMemoryBarrier ibBM{};
            ibBM.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            ibBM.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            ibBM.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ibBM.image = bmNext->image;
            ibBM.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ibBM.subresourceRange.baseMipLevel = 0;
            ibBM.subresourceRange.levelCount = 1;
            ibBM.subresourceRange.baseArrayLayer = 0;
            ibBM.subresourceRange.layerCount = 1;
            ibBM.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            ibBM.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                1, &mb,
                0, nullptr,
                1, &ibBM
            );
            bmNextLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }

    VkClearValue clear{};
    clear.color = { {0,0,0,1} };

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = renderPass;
    begin.framebuffer = framebuffers[imageIndex];
    begin.renderArea.offset = { 0, 0 };
    begin.renderArea.extent = swapchain->swapchainExtent;
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);


    if (params.mode == Mode::Mix && params.openMixEnhance) {

        if (params.enableBlurMap) {
            // ---------------- FIX #2 ----------------
            // set0/set1/set2 all must use opt, NOT 0
            pipeline->bind(
                cmd,
                0, // frameIndex fixed (opt already encodes safeFrame)
                Pipeline::DescriptorOption({ opt, opt, opt })
            );
        }
        else {
            pipeline->bind(
                cmd,
                0,
                Pipeline::DescriptorOption({ opt, opt })
            );
        }
    }
    else {
        // only set0
        pipeline->bind(
            cmd,
            0,
            Pipeline::DescriptorOption(opt)
        );
    }

    PushConst pc{};
    pc.presentMode = int(params.mode);
    pc.mixOp = params.mixOp;
    pc.mixFactor = params.mixFactor;
    pc.alphaPow = 2.2f;
    pc.featherRange = 0.06f;
    pc.depthEps = 5e-4f;
    pc.useMinDepthA = 1;
    pc.styleLock = params.styleLock;

    vkCmdPushConstants(
        cmd,
        pipeline->getPipelineLayout(),
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(PushConst),
        &pc
    );

    VkDeviceSize offset = 0;
    VkBuffer vb = fsTriangleVB->buffer;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);

    {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = swapImage;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.baseMipLevel = 0;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.baseArrayLayer = 0;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = 0;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &b
        );
        swapImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    if (params.mode == Mode::Mix && params.openMixEnhance) {
        if (!autoMatchPing.empty()) {
            autoMatchPing[safeFrame] ^= 1u;
        }
    }
}