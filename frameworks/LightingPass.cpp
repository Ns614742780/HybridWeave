#include "LightingPass.h"
#include "IGBufferProvider.h"
#include "IblEnvPass.h"
#include "Shader.h"
#include "QueryManager.h"
#include "Image.h"

#include <stdexcept>
#include <spdlog/spdlog.h>

LightingPass::LightingPass(const RenderGlobalResources& r,
    IGBufferProvider* gbufferSource,
	IblEnvPass* iblSource,
    int debugView_)
    : global(r)
    , context(r.context)
    , swapchain(r.swapchain)
	, queryManager(r.queryManager)
    , gbufferPass(gbufferSource)
	, iblEnvPass(iblSource)
    , debugView(debugView_)
{
    if (!context) throw std::runtime_error("LightingPass: context is null");
    if (!swapchain) throw std::runtime_error("LightingPass: swapchain is null");
    if (!gbufferPass) throw std::runtime_error("LightingPass: gbufferPass is null");
}

LightingPass::~LightingPass()
{
    destroyPipelineAndDescriptors();
    destroyFullscreenTriangleVB();
    destroySampler();
    destroyRenderPassAndFramebuffers();
	destroyOffscreenColor();
}

void LightingPass::initialize()
{
    spdlog::debug("LightingPass::initialize()");
    createSampler();
    createFullscreenTriangleVB();

	createOffscreenColor();
    createRenderPassAndFramebuffers();
    createPipelineAndDescriptors();
}

void LightingPass::onSwapchainResized()
{
    spdlog::debug("LightingPass::onSwapchainResized()");
	vkDeviceWaitIdle(context->device);

    destroyPipelineAndDescriptors();
    destroyRenderPassAndFramebuffers();
	destroyOffscreenColor();

	destroySampler();
	createSampler();

    createOffscreenColor();
    createRenderPassAndFramebuffers();
    createPipelineAndDescriptors();
}

void LightingPass::createOffscreenColor()
{
    const VkExtent2D extent = swapchain->swapchainExtent;
    const size_t imageCount = swapchain->swapchainImages.size();

    lightingColor.clear();
    lightingColor.resize(imageCount);
    lightingColorLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);

    auto makeImage2D = [&](VkFormat fmt) -> std::shared_ptr<Image>
        {
            VkImageCreateInfo img{};
            img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img.imageType = VK_IMAGE_TYPE_2D;
            img.extent = { extent.width, extent.height, 1 };
            img.mipLevels = 1;
            img.arrayLayers = 1;
            img.format = fmt;
            img.tiling = VK_IMAGE_TILING_OPTIMAL;
            img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            img.samples = VK_SAMPLE_COUNT_1_BIT;
            img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

            VmaAllocationCreateInfo ainfo{};
            ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            if (vmaCreateImage(context->allocator, &img, &ainfo, &image, &alloc, nullptr) != VK_SUCCESS) {
                throw std::runtime_error("LightingPass: failed to create offscreen image");
            }

            VkImageViewCreateInfo view{};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = fmt;
            view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view.subresourceRange.baseMipLevel = 0;
            view.subresourceRange.levelCount = 1;
            view.subresourceRange.baseArrayLayer = 0;
            view.subresourceRange.layerCount = 1;

            VkImageView imageView = VK_NULL_HANDLE;
            if (vkCreateImageView(context->device, &view, nullptr, &imageView) != VK_SUCCESS) {
                vmaDestroyImage(context->allocator, image, alloc);
                throw std::runtime_error("LightingPass: failed to create offscreen image view");
            }

            return std::make_shared<Image>(image, imageView, fmt, extent, VK_NULL_HANDLE, alloc);
        };

    for (size_t i = 0; i < imageCount; ++i) {
        lightingColor[i] = makeImage2D(lightingFormat);
    }
}

void LightingPass::destroyOffscreenColor()
{
    if (!context) return;

    for (auto& img : lightingColor) {
        if (!img) continue;

        if (img->imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(context->device, img->imageView, nullptr);
            img->imageView = VK_NULL_HANDLE;
        }
        if (img->image != VK_NULL_HANDLE) {
            vmaDestroyImage(context->allocator, img->image, img->allocation);
            img->image = VK_NULL_HANDLE;
            img->allocation = VK_NULL_HANDLE;
        }
        img.reset();
    }

    lightingColor.clear();
    lightingColorLayouts.clear();
}

void LightingPass::update(float)
{
}

void LightingPass::createSampler()
{
    if (envSampler != VK_NULL_HANDLE && gbufferSampler != VK_NULL_HANDLE && materialSampler != VK_NULL_HANDLE) {
        return;
	}
    if (envSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samp{};
        samp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samp.magFilter = VK_FILTER_LINEAR;
        samp.minFilter = VK_FILTER_LINEAR;
        samp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samp.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samp.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samp.minLod = 0.0f;
        samp.maxLod = (iblEnvPass && iblEnvPass->getPrefilterMipLevels() > 0)
            ? float(iblEnvPass->getPrefilterMipLevels() - 1)
            : 0.0f;
        samp.anisotropyEnable = VK_TRUE;
        samp.maxAnisotropy = 8.0f;

        if (vkCreateSampler(context->device, &samp, nullptr, &envSampler) != VK_SUCCESS) {
            throw std::runtime_error("LightingPass: failed to create sampler");
        }
    }
    if( gbufferSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo gi{};
        gi.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        gi.magFilter = VK_FILTER_LINEAR;
        gi.minFilter = VK_FILTER_LINEAR;
        gi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        gi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        gi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        gi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

		gi.minLod = 0.0f;
		gi.maxLod = 0.0f;
        gi.mipLodBias = 0.0f;

        gi.anisotropyEnable = VK_FALSE;
        gi.maxAnisotropy = 1.0f;
        
        if (vkCreateSampler(context->device, &gi, nullptr, &gbufferSampler) != VK_SUCCESS) {
            throw std::runtime_error("LightingPass: failed to create sampler");
        }
	}
    if(materialSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_NEAREST;
        info.minFilter = VK_FILTER_NEAREST;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        if (vkCreateSampler(context->device, &info, nullptr, &materialSampler) != VK_SUCCESS) {
			throw std::runtime_error("LightingPass: failed to create material sampler");
        }
	}

    
}

void LightingPass::destroySampler()
{
    if (envSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, envSampler, nullptr);
        envSampler = VK_NULL_HANDLE;
    }
    if (gbufferSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, gbufferSampler, nullptr);
        gbufferSampler = VK_NULL_HANDLE;
	}
    if (materialSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, materialSampler, nullptr);
        materialSampler = VK_NULL_HANDLE;
	}
}

void LightingPass::createFullscreenTriangleVB()
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
        "LightingPass Fullscreen Triangle VB"
    );

    fsTriangleVB->upload(verts, static_cast<uint32_t>(sizeof(verts)));
}

void LightingPass::destroyFullscreenTriangleVB()
{
    fsTriangleVB.reset();
}

void LightingPass::createRenderPassAndFramebuffers()
{
    const size_t imageCount = swapchain->swapchainImages.size();
    const VkExtent2D extent = swapchain->swapchainExtent;

    // 1) RenderPass: single color attachment (OFFSCREEN)
    VkAttachmentDescription color{};
    color.format = lightingFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency deps[2]{};
    // external -> subpass
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	// subpass -> external provide for other passes to read
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 1;
    rp.pAttachments = &color;
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 2;
    rp.pDependencies = deps;

    if (vkCreateRenderPass(context->device, &rp, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("LightingPass: failed to create offscreen render pass");
    }

    // 2) Framebuffers: one per swapchain imageIndex, but attachment is lightingColor[i]
    framebuffers.clear();
    framebuffers.resize(imageCount, VK_NULL_HANDLE);

    for (size_t i = 0; i < imageCount; ++i) {
        VkImageView views[1] = { lightingColor[i]->imageView };

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = renderPass;
        fb.attachmentCount = 1;
        fb.pAttachments = views;
        fb.width = extent.width;
        fb.height = extent.height;
        fb.layers = 1;

        if (vkCreateFramebuffer(context->device, &fb, nullptr, &framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("LightingPass: failed to create offscreen framebuffer");
        }
    }
}

void LightingPass::destroyRenderPassAndFramebuffers()
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
}

void LightingPass::createPipelineAndDescriptors()
{
    // ---------------------------------------------------------------------
    // 1) Descriptor set (per-frame, updated per imageIndex in record())
    // ---------------------------------------------------------------------
    gbufferSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);

	// use GBuffer[0] images as placeholders (for layout/build only)
    const auto& gbi0 = gbufferPass->getGBufferImages()[0];

    gbufferSet->bindCombinedImageSamplerToDescriptorSet(
        0, VK_SHADER_STAGE_FRAGMENT_BIT, gbi0.albedoRough, gbufferSampler);
    gbufferSet->bindCombinedImageSamplerToDescriptorSet(
        1, VK_SHADER_STAGE_FRAGMENT_BIT, gbi0.normalMetal, gbufferSampler);
    gbufferSet->bindCombinedImageSamplerToDescriptorSet(
        2, VK_SHADER_STAGE_FRAGMENT_BIT, gbi0.worldPos, gbufferSampler);
    gbufferSet->bindCombinedImageSamplerToDescriptorSet(
        3, VK_SHADER_STAGE_FRAGMENT_BIT, gbi0.emissiveAO, gbufferSampler);
    gbufferSet->bindCombinedImageSamplerToDescriptorSet(
        4, VK_SHADER_STAGE_FRAGMENT_BIT, gbi0.sheenColorRough, gbufferSampler);
    gbufferSet->bindCombinedImageSamplerToDescriptorSet(
        5, VK_SHADER_STAGE_FRAGMENT_BIT, gbi0.material, materialSampler);
    gbufferSet->bindCombinedImageSamplerToDescriptorSet(
        6, VK_SHADER_STAGE_FRAGMENT_BIT, gbi0.depth, gbufferSampler);

    gbufferSet->build();

    lightingSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);

    lightingSet->bindBufferToDescriptorSet(
        0,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        global.lightUboBuffer
    );

    if (iblEnvPass &&
        iblEnvPass->getEnvCubemapImage() &&
        iblEnvPass->getIrradianceCubemapImage() &&
        iblEnvPass->getEnvCubemapSampler() != VK_NULL_HANDLE &&
		iblEnvPass->getIrradianceCubemapSampler() != VK_NULL_HANDLE
        )
    {
        lightingSet->bindCombinedImageSamplerRaw(
            1,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            iblEnvPass->getEnvCubemapImage()->imageView,
            iblEnvPass->getEnvCubemapSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        lightingSet->bindCombinedImageSamplerRaw(
            2, VK_SHADER_STAGE_FRAGMENT_BIT,
            iblEnvPass->getIrradianceCubemapImage()->imageView,
            iblEnvPass->getIrradianceCubemapSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    else {
        auto dummyImg = gbufferPass->getGBufferImages()[0].albedoRough;
        auto dummySampler = envSampler;
        lightingSet->bindCombinedImageSamplerRaw(
            1,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            dummyImg->imageView,
            dummySampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        lightingSet->bindCombinedImageSamplerRaw(
            2, VK_SHADER_STAGE_FRAGMENT_BIT,
            dummyImg->imageView, envSampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    // --- Specular IBL (prefiltered env) ---
    if (iblEnvPass &&
        iblEnvPass->getPrefilterCubemapImage() &&
        iblEnvPass->getPrefilterCubemapSampler() != VK_NULL_HANDLE)
    {
        lightingSet->bindCombinedImageSamplerRaw(
            3,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            iblEnvPass->getPrefilterCubemapImage()->imageView,
            iblEnvPass->getPrefilterCubemapSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    else {
        // fallback: use env cubemap
        auto dummyImg = gbufferPass->getGBufferImages()[0].albedoRough;
        lightingSet->bindCombinedImageSamplerRaw(
            3,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            dummyImg->imageView,
            envSampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    // --- BRDF LUT ---
    if (iblEnvPass &&
        iblEnvPass->getBrdfLutImage() &&
        iblEnvPass->getBrdfLutSampler() != VK_NULL_HANDLE)
    {
        lightingSet->bindCombinedImageSamplerRaw(
            4,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            iblEnvPass->getBrdfLutImage()->imageView,
            iblEnvPass->getBrdfLutSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    else {
        // fallback (won't be used, but keeps layout valid)
        auto dummyImg = gbufferPass->getGBufferImages()[0].albedoRough;
        lightingSet->bindCombinedImageSamplerRaw(
            4,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            dummyImg->imageView,
            envSampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    // --- Sheen Specular IBL (Charlie prefiltered env) ---
    if (iblEnvPass &&
        iblEnvPass->getSheenPrefilterCubemapImage() &&
        iblEnvPass->getSheenPrefilterCubemapSampler() != VK_NULL_HANDLE)
    {
        lightingSet->bindCombinedImageSamplerRaw(
            5, // NEW binding
            VK_SHADER_STAGE_FRAGMENT_BIT,
            iblEnvPass->getSheenPrefilterCubemapImage()->imageView,
            iblEnvPass->getSheenPrefilterCubemapSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    else
    {
        // fallback: keep descriptor valid
        auto dummyImg = gbufferPass->getGBufferImages()[0].albedoRough;
        lightingSet->bindCombinedImageSamplerRaw(
            5,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            dummyImg->imageView,
            envSampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    // --- Sheen DFG LUT (Charlie + Neubelt) ---
    if (iblEnvPass &&
        iblEnvPass->getSheenLutImage() &&
        iblEnvPass->getSheenLutSampler() != VK_NULL_HANDLE)
    {
        lightingSet->bindCombinedImageSamplerRaw(
            6, // NEW binding
            VK_SHADER_STAGE_FRAGMENT_BIT,
            iblEnvPass->getSheenLutImage()->imageView,
            iblEnvPass->getSheenLutSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    else
    {
        // fallback: keep layout valid
        auto dummyImg = gbufferPass->getGBufferImages()[0].albedoRough;
        lightingSet->bindCombinedImageSamplerRaw(
            6,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            dummyImg->imageView,
            envSampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    lightingSet->build();

    uniformSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);

    uniformSet->bindBufferToDescriptorSet(
        0,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        global.uniformBuffer
    );

	uniformSet->build();

    // 2) Graphics pipeline
    pipeline = std::make_shared<GraphicsPipeline>(context);

    pipeline->setDepthTest(false);
    pipeline->addDescriptorSet(0, gbufferSet);
    pipeline->addDescriptorSet(1, lightingSet);
	pipeline->addDescriptorSet(2, uniformSet);

    pipeline->addPushConstant(
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(PushConst)
    );

    pipeline->setRenderPass(renderPass, 0);
    pipeline->setExtent(swapchain->swapchainExtent);

    // Vertex input for fullscreen triangle: vec2 pos + vec2 uv
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
    auto frag = std::make_shared<Shader>(context, "lighting_frag");
    pipeline->setShaders(vert, frag);

    pipeline->setCullMode(VK_CULL_MODE_NONE);
    pipeline->setFrontFace(VK_FRONT_FACE_CLOCKWISE);

    // IMPORTANT:
    // We need to choose descriptor option by imageIndex, not always 0.
    // Your Pipeline::DescriptorOption supports multiple options; we pass it at bind time.
    // (See record()).

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

void LightingPass::destroyPipelineAndDescriptors()
{
	gbufferSet.reset();
    lightingSet.reset();
	uniformSet.reset();
    pipeline.reset();
}

void LightingPass::record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex)
{
    VkImage image = lightingColor[imageIndex]->image;
    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex,
            cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            "lighting_total_start"
        );
    }

    auto imageBarrier = [&](VkImageLayout oldLayout, VkImageLayout newLayout,
        VkAccessFlags srcAccess, VkAccessFlags dstAccess,
        VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
        {
            if (oldLayout == newLayout) return;

            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = oldLayout;
            b.newLayout = newLayout;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = image;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.baseMipLevel = 0;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.baseArrayLayer = 0;
            b.subresourceRange.layerCount = 1;
            b.srcAccessMask = srcAccess;
            b.dstAccessMask = dstAccess;

            vkCmdPipelineBarrier(
                cmd,
                srcStage, dstStage,
                0,
                0, nullptr,
                0, nullptr,
                1, &b
            );
        };

    // 1) old -> COLOR_ATTACHMENT_OPTIMAL
    {
        VkImageLayout oldLayout = lightingColorLayouts[imageIndex];

        VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags srcAccess = 0;

        if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            srcAccess = VK_ACCESS_SHADER_READ_BIT;
        }

        imageBarrier(
            oldLayout,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            srcAccess,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            srcStage,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        );

        lightingColorLayouts[imageIndex] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // 2) Begin render pass (write lighting)
    VkClearValue clear{};
    clear.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = renderPass;
    begin.framebuffer = framebuffers[imageIndex];
    begin.renderArea.offset = { 0, 0 };
    begin.renderArea.extent = swapchain->swapchainExtent;
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex,
            cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            "lighting_pass_start"
        );
    }
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);

    // === update descriptor images for this frame ===
    const auto& gbuf = gbufferPass->getGBufferImages()[imageIndex];

    gbufferSet->updateCombinedImageSampler(
        0, frameIndex, gbuf.albedoRough, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gbufferSet->updateCombinedImageSampler(
        1, frameIndex, gbuf.normalMetal, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gbufferSet->updateCombinedImageSampler(
        2, frameIndex, gbuf.worldPos, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gbufferSet->updateCombinedImageSampler(
        3, frameIndex, gbuf.emissiveAO, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gbufferSet->updateCombinedImageSampler(
        4, frameIndex, gbuf.sheenColorRough, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gbufferSet->updateCombinedImageSampler(
        5, frameIndex, gbuf.material, materialSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gbufferSet->updateCombinedImageSampler(
        6, frameIndex, gbuf.depth, gbufferSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    pipeline->bind(
        cmd,
        static_cast<uint8_t>(frameIndex),
        Pipeline::DescriptorOption(static_cast<uint32_t>(frameIndex))
    );

    PushConst pc{};
    pc.debugView = debugView;

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
    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex,
            cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            "lighting_pass_end"
        );
    }

    // 3) renderpass's finalLayout trans all to SHADER_READ_ONLY_OPTIMAL
    lightingColorLayouts[imageIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex,
            cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            "lighting_total_end"
        );
    }
}
