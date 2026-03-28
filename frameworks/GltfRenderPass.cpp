#include "GltfRenderPass.h"

#include <stdexcept>
#include <spdlog/spdlog.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Shader.h"
#include "Camera.h"
#include "QueryManager.h"
#include <iostream>

static void generateMipmaps(
    VkCommandBuffer cmd,
    VkImage image,
    uint32_t width,
    uint32_t height,
    uint32_t mipLevels)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipW = (int32_t)width;
    int32_t mipH = (int32_t)height;

    barrier.subresourceRange.baseMipLevel = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        // --------- mip i: UNDEFINED -> TRANSFER_DST ----------
        barrier.subresourceRange.baseMipLevel = i;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // --------- blit (i-1 src -> i dst) ----------
        VkImageBlit blit{};
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { mipW, mipH, 1 };
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;

        int32_t nextW = std::max(mipW / 2, 1);
        int32_t nextH = std::max(mipH / 2, 1);

        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { nextW, nextH, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(
            cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit,
            VK_FILTER_LINEAR);

        // --------- mip(i-1): TRANSFER_SRC -> SHADER_READ ----------
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // --------- prepare next src mip i (TRANSFER_DST -> TRANSFER_SRC) ----------
        if (i < mipLevels - 1)
        {
            barrier.subresourceRange.baseMipLevel = i;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(
                cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        mipW = nextW;
        mipH = nextH;
    }

    // last mip: TRANSFER_DST -> SHADER_READ
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}


GltfRenderPass::GltfRenderPass(const RenderGlobalResources& r, const std::string& path)
    : global(r)
    , context(r.context)
    , swapchain(r.swapchain)
    , scenePath(path)
	, uniformBuffer(r.uniformBuffer)
	, camera(r.camera)
	, queryManager(r.queryManager)
{
    if (!context) throw std::runtime_error("GltfRenderPass: context is null");
    if (!swapchain) throw std::runtime_error("GltfRenderPass: swapchain is null");
}

GltfRenderPass::~GltfRenderPass()
{
    destroyPipelineAndDescriptors();

    if (dummySampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, dummySampler, nullptr);
        dummySampler = VK_NULL_HANDLE;
    }

    if (dummyWhiteImage) {
        vkDestroyImageView(context->device, dummyWhiteImage->imageView, nullptr);
        dummyWhiteImage.reset();
    }

    if (dummyImage != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, dummyImage, dummyImageAlloc);
        dummyImage = VK_NULL_HANDLE;
        dummyImageAlloc = VK_NULL_HANDLE;
    }

    destroyRenderPassAndFramebuffers();

    if (loader) {
        loader->destroy();
        loader.reset();
    }
}

void GltfRenderPass::initialize()
{
    spdlog::debug("GltfRenderPass::initialize() scene={}", scenePath);

    loader = std::make_unique<GltfLoaderVulkan>(context);
    GltfLoaderVulkan::RootTransform t;

    // bicycle
	// if you want to transform the model, you can change the values below or set them to identity
    t.translate = { -2.5f, 2.0f, -1.0f };
    t.scale = { 1.0f, 1.0f, 1.0f }; 
    t.rotate = glm::angleAxis(glm::radians(180.0f), glm::vec3(1, 0, 0));

    loader->setRootTransform(t);
    if (!loader->loadModel(scenePath)) {
        throw std::runtime_error("GltfRenderPass: failed to load gltf model");
    }
    loader->dumpSummary();

    createRenderPassAndFramebuffers();
    createPipelineAndDescriptors();
	initialized = true;
}

void GltfRenderPass::onSwapchainResized()
{
    spdlog::debug("GltfRenderPass::onSwapchainResized()");

    vkDeviceWaitIdle(context->device);

    destroyPipelineAndDescriptors();
    destroyRenderPassAndFramebuffers();

    createRenderPassAndFramebuffers();
    createPipelineAndDescriptors();
}

void GltfRenderPass::createRenderPassAndFramebuffers()
{
    const VkExtent2D extent = swapchain->swapchainExtent;
	const size_t imageCount = swapchain->swapchainImages.size();

	gbufferImages.clear();
	gbufferImages.resize(imageCount);

	// culculate mip levels for gbuffer textures
    auto calcMipLevels = [](uint32_t w, uint32_t h) {
        return static_cast<uint32_t>(
            std::floor(std::log2(std::max(w, h)))
            ) + 1;
        };
    gbufferMipLevels =
        calcMipLevels(extent.width, extent.height);

    // ---------- helper: create image + view + Image(shared_ptr) ----------
    auto createAttachment = [&](VkFormat format,
        VkImageUsageFlags usage,
        VkImageAspectFlags aspectMask,
        uint32_t mipLevels) -> std::shared_ptr<Image>
        {
            VkImageCreateInfo img{};
            img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img.imageType = VK_IMAGE_TYPE_2D;
            img.extent = { extent.width, extent.height, 1 };
            img.mipLevels = mipLevels;
            img.arrayLayers = 1;
            img.format = format;
            img.tiling = VK_IMAGE_TILING_OPTIMAL;
            img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            img.samples = VK_SAMPLE_COUNT_1_BIT;
            img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img.usage =
                usage |
                VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;

            VmaAllocationCreateInfo ainfo{};
            ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            vmaCreateImage(context->allocator, &img, &ainfo, &image, &alloc, nullptr);

            VkImageViewCreateInfo viewAll{};
            viewAll.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewAll.image = image;
            viewAll.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewAll.format = format;
            viewAll.subresourceRange.aspectMask = aspectMask;
            viewAll.subresourceRange.baseMipLevel = 0;
            viewAll.subresourceRange.levelCount = mipLevels;
            viewAll.subresourceRange.baseArrayLayer = 0;
            viewAll.subresourceRange.layerCount = 1;

            VkImageView allMipView = VK_NULL_HANDLE;
            vkCreateImageView(context->device, &viewAll, nullptr, &allMipView);

            auto out = std::make_shared<Image>(
                image, allMipView, format, extent, VK_NULL_HANDLE, alloc);

            VkImageViewCreateInfo viewMip0 = viewAll;
            viewMip0.subresourceRange.levelCount = 1;
            vkCreateImageView(context->device, &viewMip0, nullptr, &out->mip0View);

            return out;
        };

    for(size_t i = 0; i < imageCount; ++i) {
        gbufferImages[i].albedoRough =
            createAttachment(albedoRoughFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                gbufferMipLevels);
        gbufferImages[i].normalMetal =
            createAttachment(normalMetalFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                gbufferMipLevels);
        gbufferImages[i].worldPos =
            createAttachment(worldPosFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                1);
        gbufferImages[i].emissiveAO =
            createAttachment(emissiveAOFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                gbufferMipLevels);
        gbufferImages[i].sheenColorRough =
            createAttachment(sheenFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
				1);
        gbufferImages[i].material =
            createAttachment(materialFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                1);
        gbufferImages[i].drawId =
            createAttachment(drawIdFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                1);
        gbufferImages[i].depth =
            createAttachment(depthFormat,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT,
                1);
	}

    // ---------- render pass: 7 MRT + depth ----------
    VkAttachmentDescription att[8]{};

    // G0 albedoRough
    att[0].format = albedoRoughFormat;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // G1 normalMetal
    att[1] = att[0];
    att[1].format = normalMetalFormat;

    // G2 worldPos
    att[2] = att[0];
    att[2].format = worldPosFormat;

    // G3 emissiveAO
    att[3] = att[0];
    att[3].format = emissiveAOFormat;

	// G4 material
    att[4] = att[0];
    att[4].format = sheenFormat;

	// G5 material
	att[5] = att[0];
	att[5].format = materialFormat;

	// G6 drawId
	att[6] = att[0];
	att[6].format = drawIdFormat;

    // Depth
    att[7].format = depthFormat;
    att[7].samples = VK_SAMPLE_COUNT_1_BIT;
    att[7].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[7].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[7].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[7].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[7].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[7].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRefs[7]{};
    for (int k = 0; k < 7; ++k) {
        colorRefs[k].attachment = k;
        colorRefs[k].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    VkAttachmentReference depthRef{};
    depthRef.attachment = 7;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 7;
    subpass.pColorAttachments = colorRefs;
    subpass.pDepthStencilAttachment = &depthRef;

    // Dependencies: make color writes visible to later shader reads (for lighting/debug pass)
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
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 8;
    rp.pAttachments = att;
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 2;
    rp.pDependencies = deps;

    if (vkCreateRenderPass(context->device, &rp, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("GltfRenderPass: failed to create GBuffer render pass");
    }

    framebuffers.clear();
    framebuffers.resize(imageCount, VK_NULL_HANDLE);

    for (size_t i = 0; i < imageCount; ++i) {
        VkImageView views[8] = {
            gbufferImages[i].albedoRough->mip0View,
            gbufferImages[i].normalMetal->mip0View,
            gbufferImages[i].worldPos->imageView,
            gbufferImages[i].emissiveAO->mip0View,
			gbufferImages[i].sheenColorRough->mip0View,
            gbufferImages[i].material->mip0View,
            gbufferImages[i].drawId->mip0View,
            gbufferImages[i].depth->imageView
        };

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = renderPass;
        fb.attachmentCount = 8;
        fb.pAttachments = views;
        fb.width = extent.width;
        fb.height = extent.height;
        fb.layers = 1;

        if (vkCreateFramebuffer(context->device, &fb, nullptr, &framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("GltfRenderPass: failed to create GBuffer framebuffer");
        }
    }
}

void GltfRenderPass::destroyRenderPassAndFramebuffers()
{
    if (!context) return;

    for (auto& gbi : gbufferImages) {
        auto destroyAttachment = [&](std::shared_ptr<Image>& img) {
            if (!img) return;

            if (img->mip0View != VK_NULL_HANDLE) {
                vkDestroyImageView(context->device, img->mip0View, nullptr);
                img->mip0View = VK_NULL_HANDLE;
            }
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
        };

        destroyAttachment(gbi.albedoRough);
        destroyAttachment(gbi.normalMetal);
        destroyAttachment(gbi.worldPos);
        destroyAttachment(gbi.emissiveAO);
		destroyAttachment(gbi.sheenColorRough);
        destroyAttachment(gbi.material);
		destroyAttachment(gbi.drawId);
        destroyAttachment(gbi.depth);
	}
    gbufferImages.clear();

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

void GltfRenderPass::createPipelineAndDescriptors()
{
    // Wrap loader textures into Image wrappers for DescriptorSet
    textureImageWrappers.clear();
    {
        const auto& tex = loader->getTextures();
        textureImageWrappers.reserve(tex.size());

        for (const auto& t : tex) {
            VkExtent2D dummy{ 0, 0 };
            textureImageWrappers.push_back(std::make_shared<Image>(t.image, t.view, t.format, dummy));
        }
    }

    // Ensure we have a valid dummy texture for the "no textures" case
    if (!dummyWhiteImage) {
        createDummyWhiteTexture();
    }
    
	// Ensure we have a uniform buffer

    // -------------------------------------------------------------------------
    // Descriptor set: set0
    // binding0: UBO
    // binding1: sampler2D[] array (or 1-element dummy array if scene has no textures)
    // -------------------------------------------------------------------------
    mainSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);

    mainSet->bindBufferToDescriptorSet(
        0,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        uniformBuffer
    );

    {
        const auto& tex = loader->getTextures();

        if (!tex.empty()) {
            std::vector<VkSampler> samplers;
            samplers.reserve(tex.size());
            for (const auto& t : tex) samplers.push_back(t.sampler);

            mainSet->bindCombinedImageSamplerArrayToDescriptorSet(
                1,
                VK_SHADER_STAGE_FRAGMENT_BIT,
                textureImageWrappers,
                samplers,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
        }
        else {
            std::vector<std::shared_ptr<Image>> imgs = { dummyWhiteImage };
            std::vector<VkSampler> samplers = { dummySampler };

            mainSet->bindCombinedImageSamplerArrayToDescriptorSet(
                1,
                VK_SHADER_STAGE_FRAGMENT_BIT,
                imgs,
                samplers,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
        }
    }

    mainSet->build();

    // Graphics pipeline
    pipeline = std::make_shared<GraphicsPipeline>(context);
	pipeline->setDepthTest(true);
    pipeline->setFlippedViewport(false);
	pipeline->setColorAttachmentCount(7); // MRT7
    pipeline->addDescriptorSet(0, mainSet);

    // push constants
    pipeline->addPushConstant(
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(PushConst)
    );

    pipeline->setRenderPass(renderPass, 0);
    pipeline->setExtent(swapchain->swapchainExtent);

    // Vertex input: 12 floats = 48 bytes stride
    GraphicsPipeline::VertexInputDesc vi{};
    vi.binding.binding = 0;
    vi.binding.stride = sizeof(float) * 12;
    vi.binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    vi.attributes.resize(4);
    vi.attributes[0].location = 0;
    vi.attributes[0].binding = 0;
    vi.attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vi.attributes[0].offset = 0;

    vi.attributes[1].location = 1;
    vi.attributes[1].binding = 0;
    vi.attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    vi.attributes[1].offset = sizeof(float) * 3;

    vi.attributes[2].location = 2;
    vi.attributes[2].binding = 0;
    vi.attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
    vi.attributes[2].offset = sizeof(float) * 6;

    vi.attributes[3].location = 3;
    vi.attributes[3].binding = 0;
    vi.attributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    vi.attributes[3].offset = sizeof(float) * 8;

    pipeline->setVertexInput(vi);

    // Shaders
    auto vert = std::make_shared<Shader>(context, "gltf_vert");
    auto frag = std::make_shared<Shader>(context, "gltf_frag");
    pipeline->setShaders(vert, frag);

    pipeline->setFrontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipeline->setCullMode(VK_CULL_MODE_BACK_BIT);

    std::vector<VkPipelineColorBlendAttachmentState> blends(7);

    // 0~4£∫∆’Õ® RGBA GBuffer
    for (int i = 0; i < 5; ++i) {
        blends[i].blendEnable = VK_FALSE;
        blends[i].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
    }

    // 5£∫material (R32_UINT)
    blends[5].blendEnable = VK_FALSE;
    blends[5].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;

	// 6£∫drawId (R32_UINT)
	blends[6].blendEnable = VK_FALSE;
	blends[6].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;

    pipeline->setColorBlendAttachments(blends);

    pipeline->build();
}

void GltfRenderPass::destroyPipelineAndDescriptors()
{
    pipeline.reset();
    mainSet.reset();
    textureImageWrappers.clear();
}

void GltfRenderPass::createDummyWhiteTexture()
{
    if (dummyWhiteImage) return; // already created

    auto device = context->device;

    VkImageCreateInfo img{};
    img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img.imageType = VK_IMAGE_TYPE_2D;
    img.format = VK_FORMAT_R8G8B8A8_UNORM;
    img.extent = { 1, 1, 1 };
    img.mipLevels = 1;
    img.arrayLayers = 1;
    img.samples = VK_SAMPLE_COUNT_1_BIT;
    img.tiling = VK_IMAGE_TILING_OPTIMAL;
    img.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateImage(
        context->allocator,
        &img,
        &alloc,
        &dummyImage,
        &dummyImageAlloc,
        nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create dummy white image");
    }

    uint32_t white = 0xffffffff;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;

    VkBufferCreateInfo buf{};
    buf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf.size = sizeof(uint32_t);
    buf.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo bufAlloc{};
    bufAlloc.usage = VMA_MEMORY_USAGE_AUTO;
    bufAlloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    vmaCreateBuffer(
        context->allocator,
        &buf,
        &bufAlloc,
        &stagingBuffer,
        &stagingAlloc,
        nullptr);

    void* mapped = nullptr;
    vmaMapMemory(context->allocator, stagingAlloc, &mapped);
    memcpy(mapped, &white, sizeof(uint32_t));
    vmaUnmapMemory(context->allocator, stagingAlloc);

    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    // UNDEFINED -> TRANSFER_DST
    VkImageMemoryBarrier b1{};
    b1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.srcAccessMask = 0;
    b1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.image = dummyImage;
    b1.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b1.subresourceRange.levelCount = 1;
    b1.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &b1);

    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = { 1, 1, 1 };

    vkCmdCopyBufferToImage(
        cmd,
        stagingBuffer,
        dummyImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copy);

    // TRANSFER_DST -> SHADER_READ
    VkImageMemoryBarrier b2 = b1;
    b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &b2);

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::GRAPHICS);

    // staging cleanup
    vmaDestroyBuffer(context->allocator, stagingBuffer, stagingAlloc);

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = dummyImage;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R8G8B8A8_UNORM;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;

    VkImageView imageView = VK_NULL_HANDLE;
    vkCreateImageView(device, &view, nullptr, &imageView);

    VkSamplerCreateInfo samp{};
    samp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp.magFilter = VK_FILTER_LINEAR;
    samp.minFilter = VK_FILTER_LINEAR;
    samp.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp.maxLod = 1.0f;

    vkCreateSampler(device, &samp, nullptr, &dummySampler);

    dummyWhiteImage = std::make_shared<Image>(
        dummyImage,
        imageView,
        VK_FORMAT_R8G8B8A8_UNORM,
        VkExtent2D{ 1,1 }
    );
}

uint32_t GltfRenderPass::pickTextureOptionForMesh(const GltfMeshGPU& m) const
{
    if (m.hasBaseColorTex) {
        // baseColorTex is "image index" in your loader
        return static_cast<uint32_t>(m.baseColorTex);
    }
    return fallbackTextureOption;
}

void GltfRenderPass::record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex)
{
    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            "gltf_total_start"
        );
    }
    VkClearValue clears[8]{};
	clears[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} }; // albedoRough
	clears[1].color = { {0.5f, 0.5f, 1.0f, 1.0f} }; // normalMetal
	clears[2].color = { {0.0f, 0.0f, 0.0f, 1.0f} }; // worldPos
	clears[3].color = { {0.0f, 0.0f, 0.0f, 1.0f} }; // emissiveAO
	clears[4].color = { {0.0f, 0.0f, 0.0f, 1.0f} }; // sheenColorRough
	clears[5].color = { {0.0f, 0.0f, 0.0f, 1.0f} }; // material           
    clears[6].color.uint32[0] = 0u;                 // drawId
    clears[6].color.uint32[1] = 0u;
    clears[6].color.uint32[2] = 0u;
    clears[6].color.uint32[3] = 0u;
	clears[7].depthStencil = { 1.0f, 0 };           // depth

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = renderPass;
    begin.framebuffer = framebuffers[imageIndex];
    begin.renderArea.offset = { 0, 0 };
    begin.renderArea.extent = swapchain->swapchainExtent;
    begin.clearValueCount = 8;
    begin.pClearValues = clears;

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            "gltf_gbuffer_start"
        );
    }
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);

    pipeline->bind(cmd, static_cast<uint8_t>(frameIndex), Pipeline::DescriptorOption(0u));

    const bool hasAnyTextures = !loader->getTextures().empty();

    // Draw all meshes
    const auto& items = loader->getDrawItems();
    const auto& meshes = loader->getMeshes();

    for (uint32_t drawItemIndex = 0;
        drawItemIndex < static_cast<uint32_t>(items.size());
        ++drawItemIndex) {
        const auto& di = items[drawItemIndex];
        const auto& m = meshes[di.meshGpuIndex];

        // vertex/index buffers
        VkDeviceSize offset = 0;
        VkBuffer vb = m.vertexBuffer->buffer;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
        vkCmdBindIndexBuffer(cmd, m.indexBuffer->buffer, 0, m.indexType);

        // push constants: model + factor + useTexture + texIndex
        PushConst pc{};
        pc.model = di.world;
        pc.baseColorFactor = glm::vec4(m.baseColorFactor);

        // ---- Metallic / Roughness factors ----
        pc.metallicFactor = m.metallicFactor;
        pc.roughnessFactor = m.roughnessFactor;

        // ---- Base color texture ----
        if (m.hasBaseColorTex && m.baseColorTex >= 0) {
            pc.baseColorTex = static_cast<uint32_t>(m.baseColorTex);
        }
        else {
            pc.baseColorTex = UINT32_MAX;
        }

        // ---- Metallic-Roughness texture ----
        if (m.hasMetallicRoughnessTex && m.metallicRoughnessTex >= 0) {
            pc.mrTex = static_cast<uint32_t>(m.metallicRoughnessTex);
        }
        else {
            pc.mrTex = UINT32_MAX;
        }

        pc.materialFlags = static_cast<uint32_t>(m.materialFlags);
        pc.drawId = static_cast<uint32_t>(di.drawId);
		pc._pad2 = 0;
		pc._pad3 = 0;

        pc.sheenColorRoughFactor = glm::vec4(m.sheenColorFactor, m.sheenRoughnessFactor);
        pc.sheenColorTex = (m.hasSheen && m.sheenColorTex >= 0)
            ? (uint32_t)m.sheenColorTex
            : UINT32_MAX;
        pc.sheenRoughTex = (m.hasSheen && m.sheenRoughnessTex >= 0)
            ? (uint32_t)m.sheenRoughnessTex
            : UINT32_MAX;
		pc._pad4 = 0;

        vkCmdPushConstants(
            cmd,
            pipeline->getPipelineLayout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(PushConst),
            &pc
        );

        vkCmdDrawIndexed(cmd, m.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            "gltf_gbuffer_end"
        );
    }

    generateMipmaps(cmd, gbufferImages[imageIndex].albedoRough->image,
        swapchain->swapchainExtent.width, swapchain->swapchainExtent.height, gbufferMipLevels);
    generateMipmaps(cmd, gbufferImages[imageIndex].normalMetal->image,
        swapchain->swapchainExtent.width, swapchain->swapchainExtent.height,gbufferMipLevels);
    generateMipmaps(cmd, gbufferImages[imageIndex].emissiveAO->image,
        swapchain->swapchainExtent.width, swapchain->swapchainExtent.height, gbufferMipLevels);

    if (global.enableQueryManager && queryManager) {
        queryManager->writeTimestamp(
            frameIndex, cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            "gltf_total_end"
        );
    }
}

void GltfRenderPass::update(float dt)
{
}
