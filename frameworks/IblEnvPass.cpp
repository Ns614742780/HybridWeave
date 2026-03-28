#include "IblEnvPass.h"

#include <stdexcept>
#include <vector>
#include <cstring>

#include <spdlog/spdlog.h>
#include "stb_image.h"
#include "Utils.h"

static void transitionImage(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        cmd,
        srcStage, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

static void transitionImageLayers(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t baseLayer,
    uint32_t layerCount,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = baseLayer;
    barrier.subresourceRange.layerCount = layerCount;

    vkCmdPipelineBarrier(
        cmd,
        srcStage, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

static void transitionImageSubresources(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t baseMipLevel,
    uint32_t levelCount,
    uint32_t baseLayer,
    uint32_t layerCount,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = baseMipLevel;
    barrier.subresourceRange.levelCount = levelCount;
    barrier.subresourceRange.baseArrayLayer = baseLayer;
    barrier.subresourceRange.layerCount = layerCount;

    vkCmdPipelineBarrier(
        cmd,
        srcStage, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

static uint32_t calcMipLevels(uint32_t size)
{
    uint32_t levels = 1;
    while (size > 1) {
        size >>= 1;
        levels++;
    }
    return levels;
}


IblEnvPass::IblEnvPass(const RenderGlobalResources& r, std::string path, float intensity)
    : global(r), context(r.context), hdrPath(std::move(path)), envIntensity(intensity)
{
}

IblEnvPass::~IblEnvPass()
{
    if (!context) return;

	destroySheenLutPipeline();
	destroySheenLutResources();

	destroySheenPrefilterPipeline();
	destroySheenPrefilterResources();

    destroyBrdfLutPipeline();
    destroyBrdfLutResources();

	destroyPrefilterPipeline();
	destroyPrefilterResources();

    destroyIrradiancePipeline();
    destroyConvertPipeline();

	destroyIrradianceResources();
    destroyCubemapResources();

    destroySampler();

    if (envEquirect && envEquirect->imageView) {
        vkDestroyImageView(context->device, envEquirect->imageView, nullptr);
        envEquirect->imageView = VK_NULL_HANDLE;
    }
    if (envEquirect && envEquirect->image && envEquirect->allocation) {
        vmaDestroyImage(context->allocator, envEquirect->image, envEquirect->allocation);
        envEquirect->image = VK_NULL_HANDLE;
        envEquirect->allocation = VK_NULL_HANDLE;
    }
    envEquirect.reset();
}

void IblEnvPass::initialize()
{
    if (!context) throw std::runtime_error("IblEnvPass::initialize: context is null");

    createSampler();

    if (hdrPath.empty()) {
        spdlog::warn("IblEnvPass: hdrPath empty, env lighting disabled.");
        return;
    }

    loadHDRToTexture2D(hdrPath);

    createCubemapResources();
    createConvertPipeline();
    runEquirectToCubemapOnce();

    createIrradianceResources();
    createIrradiancePipeline();
    runIrradianceConvolutionOnce();

	createPrefilterResources();
	createPrefilterPipeline();
	runPrefilterOnce();

	createBrdfLutResources();
	createBrdfLutPipeline();
	runBrdfLutOnce();

    createSheenPrefilterResources();
    createSheenPrefilterPipeline();
	runSheenPrefilterOnce();

    createSheenLutResources();
	createSheenLutPipeline();
	runSheenLutOnce();

    float mipLevelsF = float(prefilterMipLevels);
    constexpr uint32_t ENV_PARAMS_Z_OFFSET =
        offsetof(LightUBO, envParams) + offsetof(glm::vec4, z);
    global.lightUboBuffer->upload(
        &mipLevelsF,
        sizeof(float),
        ENV_PARAMS_Z_OFFSET
    );
}

void IblEnvPass::createSampler()
{
    if (envSampler != VK_NULL_HANDLE) return;

    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    info.mipLodBias = 0.0f;
    info.anisotropyEnable = VK_FALSE;
    info.maxAnisotropy = 1.0f;
    info.compareEnable = VK_FALSE;
    info.minLod = 0.0f;
    info.maxLod = 0.0f;
    info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(context->device, &info, nullptr, &envSampler) != VK_SUCCESS) {
        throw std::runtime_error("IblEnvPass::createSampler: failed");
    }
}

void IblEnvPass::destroySampler()
{
    if (envSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, envSampler, nullptr);
        envSampler = VK_NULL_HANDLE;
    }
    if (envCubeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, envCubeSampler, nullptr);
        envCubeSampler = VK_NULL_HANDLE;
	}
    if (irrCubeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, irrCubeSampler, nullptr);
        irrCubeSampler = VK_NULL_HANDLE;
	}
    if (prefilterSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, prefilterSampler, nullptr);
        prefilterSampler = VK_NULL_HANDLE;
	}
    if (brdfLutSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, brdfLutSampler, nullptr);
        brdfLutSampler = VK_NULL_HANDLE;
	}
    if (sheenPrefilterSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, sheenPrefilterSampler, nullptr);
        sheenPrefilterSampler = VK_NULL_HANDLE;
    }
    if (sheenLutSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, sheenLutSampler, nullptr);
        sheenLutSampler = VK_NULL_HANDLE;
	}
}

void IblEnvPass::createCubemapResources()
{
    destroyCubemapResources();

    envCube = std::make_shared<Image>();
    envCube->format = VK_FORMAT_R16G16B16A16_SFLOAT;
    envCube->extent = VkExtent2D{ (uint32_t)cubeSize, (uint32_t)cubeSize };

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = envCube->format;
    ici.extent = { (uint32_t)cubeSize, (uint32_t)cubeSize, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 6;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;

    ici.usage =
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;

    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(
        context->allocator, &ici, &ainfo,
        &envCube->image, &envCube->allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("IblEnvPass::createCubemapResources: vmaCreateImage failed");
    }

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = envCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format = envCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &envCube->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createCubemapResources: create 2D_ARRAY view failed");
        }
    }

    envCubeSampled = std::make_shared<Image>();
    envCubeSampled->image = envCube->image;
    envCubeSampled->format = envCube->format;
    envCubeSampled->extent = envCube->extent;
    envCubeSampled->allocation = VK_NULL_HANDLE;

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = envCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        vci.format = envCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &envCubeSampled->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createCubemapResources: create CUBE view failed");
        }
    }

    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        info.minLod = 0.0f;
        info.maxLod = 0.0f;
        info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(context->device, &info, nullptr, &envCubeSampler) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createCubemapResources: create cubemap sampler failed");
        }
    }

    envCubeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::destroyCubemapResources()
{
    if (!context) return;

    if (envCubeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, envCubeSampler, nullptr);
        envCubeSampler = VK_NULL_HANDLE;
    }

    if (envCubeSampled && envCubeSampled->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, envCubeSampled->imageView, nullptr);
        envCubeSampled->imageView = VK_NULL_HANDLE;
    }
    envCubeSampled.reset();

    if (envCube && envCube->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, envCube->imageView, nullptr);
        envCube->imageView = VK_NULL_HANDLE;
    }
    if (envCube && envCube->image != VK_NULL_HANDLE && envCube->allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, envCube->image, envCube->allocation);
        envCube->image = VK_NULL_HANDLE;
        envCube->allocation = VK_NULL_HANDLE;
    }
    envCube.reset();

    envCubeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::createIrradianceResources()
{
    destroyIrradianceResources();

    irrCube = std::make_shared<Image>();
    irrCube->format = VK_FORMAT_R16G16B16A16_SFLOAT;
    irrCube->extent = VkExtent2D{ (uint32_t)irradianceSize, (uint32_t)irradianceSize };

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = irrCube->format;
    ici.extent = { (uint32_t)irradianceSize, (uint32_t)irradianceSize, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 6;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(context->allocator, &ici, &ainfo,
        &irrCube->image, &irrCube->allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("IblEnvPass::createIrradianceResources: vmaCreateImage failed");
    }

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = irrCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format = irrCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &irrCube->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createIrradianceResources: create 2D_ARRAY view failed");
        }
    }

    irrCubeSampled = std::make_shared<Image>();
    irrCubeSampled->image = irrCube->image;
    irrCubeSampled->format = irrCube->format;
    irrCubeSampled->extent = irrCube->extent;
    irrCubeSampled->allocation = VK_NULL_HANDLE;

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = irrCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        vci.format = irrCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &irrCubeSampled->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createIrradianceResources: create CUBE view failed");
        }
    }

    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        info.minLod = 0.0f;
        info.maxLod = 0.0f;
        info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(context->device, &info, nullptr, &irrCubeSampler) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createIrradianceResources: create sampler failed");
        }
    }

    irrCubeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::destroyIrradianceResources()
{
    if (!context) return;

    if (irrCubeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(context->device, irrCubeSampler, nullptr);
        irrCubeSampler = VK_NULL_HANDLE;
    }

    if (irrCubeSampled && irrCubeSampled->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, irrCubeSampled->imageView, nullptr);
        irrCubeSampled->imageView = VK_NULL_HANDLE;
    }
    irrCubeSampled.reset();

    if (irrCube && irrCube->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, irrCube->imageView, nullptr);
        irrCube->imageView = VK_NULL_HANDLE;
    }
    if (irrCube && irrCube->image != VK_NULL_HANDLE && irrCube->allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, irrCube->image, irrCube->allocation);
        irrCube->image = VK_NULL_HANDLE;
        irrCube->allocation = VK_NULL_HANDLE;
    }
    irrCube.reset();

    irrCubeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::createIrradiancePipeline()
{
    destroyIrradiancePipeline();

    if (!envCubeSampled || envCubeSampler == VK_NULL_HANDLE) {
        throw std::runtime_error("IblEnvPass::createIrradiancePipeline: envCube not ready (need CUBE view + sampler)");
    }
    if (!irrCube) {
        throw std::runtime_error("IblEnvPass::createIrradiancePipeline: irradiance cube not created");
    }

    irradiancePipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "irradiance_convolution")
    );

    irradianceSet = std::make_shared<DescriptorSet>(context, 1);

    irradianceSet->bindCombinedImageSamplerToDescriptorSet(
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        envCubeSampled,
        envCubeSampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    irradianceSet->bindImageToDescriptorSet(
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_SHADER_STAGE_COMPUTE_BIT,
        irrCube
    );

    irradianceSet->build();

    irradiancePipeline->addDescriptorSet(0, irradianceSet);

    irradiancePipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t));

    irradiancePipeline->build();
}

void IblEnvPass::destroyIrradiancePipeline()
{
    irradianceSet.reset();
    irradiancePipeline.reset();
}

void IblEnvPass::createConvertPipeline()
{
    destroyConvertPipeline();

    if (!envEquirect || envEquirect->imageView == VK_NULL_HANDLE) {
        throw std::runtime_error("IblEnvPass::createConvertPipeline: envEquirect not ready");
    }
    if (!envCube || envCube->imageView == VK_NULL_HANDLE) {
        throw std::runtime_error("IblEnvPass::createConvertPipeline: envCube not ready");
    }

    convertPipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "equirect_to_cubemap")
    );

    convertSet = std::make_shared<DescriptorSet>(context, 1);
    convertSet->bindCombinedImageSamplerToDescriptorSet(
        0, VK_SHADER_STAGE_COMPUTE_BIT,
        envEquirect, envSampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
    convertSet->bindImageToDescriptorSet(
        1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_SHADER_STAGE_COMPUTE_BIT,
        envCube
    );
    convertSet->build();

    convertPipeline->addDescriptorSet(0, convertSet);
    convertPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t));
    convertPipeline->build();
}

void IblEnvPass::destroyConvertPipeline()
{
    convertSet.reset();
    convertPipeline.reset();
}

void IblEnvPass::createPrefilterResources()
{
    destroyPrefilterResources();

    if (!context) throw std::runtime_error("IblEnvPass::createPrefilterResources: context null");

    prefilterMipLevels = calcMipLevels((uint32_t)prefilterSize);

    prefilterCube = std::make_shared<Image>();
    prefilterCube->format = VK_FORMAT_R16G16B16A16_SFLOAT;
    prefilterCube->extent = VkExtent2D{ (uint32_t)prefilterSize, (uint32_t)prefilterSize };

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = prefilterCube->format;
    ici.extent = { (uint32_t)prefilterSize, (uint32_t)prefilterSize, 1 };
    ici.mipLevels = prefilterMipLevels;
    ici.arrayLayers = 6;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(
        context->allocator, &ici, &ainfo,
        &prefilterCube->image, &prefilterCube->allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("IblEnvPass::createPrefilterResources: vmaCreateImage failed");
    }

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = prefilterCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format = prefilterCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = prefilterMipLevels;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &prefilterCube->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createPrefilterResources: create 2D_ARRAY view failed");
        }
    }

    prefilterCubeSampled = std::make_shared<Image>();
    prefilterCubeSampled->image = prefilterCube->image;
    prefilterCubeSampled->format = prefilterCube->format;
    prefilterCubeSampled->extent = prefilterCube->extent;
    prefilterCubeSampled->allocation = VK_NULL_HANDLE;

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = prefilterCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        vci.format = prefilterCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = prefilterMipLevels;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &prefilterCubeSampled->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createPrefilterResources: create CUBE view failed");
        }
    }

    prefilterMipViews.clear();
    prefilterMipViews.resize(prefilterMipLevels);

    for (uint32_t mip = 0; mip < prefilterMipLevels; ++mip) {
        auto v = std::make_shared<Image>();
        v->image = prefilterCube->image;
        v->format = prefilterCube->format;
        v->extent = prefilterCube->extent;
        v->allocation = VK_NULL_HANDLE;

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = prefilterCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format = prefilterCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = mip;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &v->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createPrefilterResources: create per-mip 2D_ARRAY view failed");
        }

        prefilterMipViews[mip] = v;
    }

    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        info.minLod = 0.0f;
        info.maxLod = (float)(prefilterMipLevels - 1);
        info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(context->device, &info, nullptr, &prefilterSampler) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createPrefilterResources: create sampler failed");
        }
    }

    prefilterLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::destroyPrefilterResources()
{
    if (!context) return;

    for (auto& v : prefilterMipViews) {
        if (v && v->imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(context->device, v->imageView, nullptr);
            v->imageView = VK_NULL_HANDLE;
        }
    }
    prefilterMipViews.clear();

    if (prefilterCubeSampled && prefilterCubeSampled->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, prefilterCubeSampled->imageView, nullptr);
        prefilterCubeSampled->imageView = VK_NULL_HANDLE;
    }
    prefilterCubeSampled.reset();

    if (prefilterCube && prefilterCube->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, prefilterCube->imageView, nullptr);
        prefilterCube->imageView = VK_NULL_HANDLE;
    }
    if (prefilterCube && prefilterCube->image != VK_NULL_HANDLE && prefilterCube->allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, prefilterCube->image, prefilterCube->allocation);
        prefilterCube->image = VK_NULL_HANDLE;
        prefilterCube->allocation = VK_NULL_HANDLE;
    }
    prefilterCube.reset();

    prefilterMipLevels = 0;
    prefilterLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::createPrefilterPipeline()
{
    destroyPrefilterPipeline();

    if (!envCubeSampled || envCubeSampler == VK_NULL_HANDLE)
        throw std::runtime_error("createPrefilterPipeline: envCube not ready");
    if (prefilterMipViews.empty() || prefilterMipLevels == 0)
        throw std::runtime_error("createPrefilterPipeline: prefilter mip views not ready");

    prefilterPipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "prefilter_env")
    );

    prefilterSet = std::make_shared<DescriptorSet>(context, 1);

    prefilterSet->maxOptions = prefilterMipLevels;

    prefilterSet->bindCombinedImageSamplerToDescriptorSet(
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        envCubeSampled,
        envCubeSampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    for (uint32_t mip = 0; mip < prefilterMipLevels; ++mip) {
        prefilterSet->bindImageToDescriptorSet(
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_COMPUTE_BIT,
            prefilterMipViews[mip]
        );
    }

    prefilterSet->build();

    prefilterPipeline->addDescriptorSet(0, prefilterSet);

    prefilterPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, 16);

    prefilterPipeline->build();
}

void IblEnvPass::destroyPrefilterPipeline()
{
    prefilterSet.reset();
    prefilterPipeline.reset();
}

void IblEnvPass::createBrdfLutResources()
{
    destroyBrdfLutResources();

    if (!context) throw std::runtime_error("IblEnvPass::createBrdfLutResources: context null");

    brdfLut = std::make_shared<Image>();
    brdfLut->format = VK_FORMAT_R16G16_SFLOAT;
    brdfLut->extent = VkExtent2D{ (uint32_t)brdfLutSize, (uint32_t)brdfLutSize };

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = brdfLut->format;
    ici.extent = { (uint32_t)brdfLutSize, (uint32_t)brdfLutSize, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(context->allocator, &ici, &ainfo,
        &brdfLut->image, &brdfLut->allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("IblEnvPass::createBrdfLutResources: vmaCreateImage failed");
    }

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = brdfLut->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = brdfLut->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 1;

        if (vkCreateImageView(context->device, &vci, nullptr, &brdfLut->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createBrdfLutResources: create view failed");
        }
    }

    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        info.minLod = 0.0f;
        info.maxLod = 0.0f;
        info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(context->device, &info, nullptr, &brdfLutSampler) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createBrdfLutResources: create sampler failed");
        }
    }

    brdfLutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::destroyBrdfLutResources()
{
    if (!context) return;

    if (brdfLut && brdfLut->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, brdfLut->imageView, nullptr);
        brdfLut->imageView = VK_NULL_HANDLE;
    }
    if (brdfLut && brdfLut->image != VK_NULL_HANDLE && brdfLut->allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, brdfLut->image, brdfLut->allocation);
        brdfLut->image = VK_NULL_HANDLE;
        brdfLut->allocation = VK_NULL_HANDLE;
    }
    brdfLut.reset();

    brdfLutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::createBrdfLutPipeline()
{
    destroyBrdfLutPipeline();

    if (!brdfLut || brdfLut->imageView == VK_NULL_HANDLE) {
        throw std::runtime_error("IblEnvPass::createBrdfLutPipeline: brdfLut not ready");
    }

    brdfLutPipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "brdf_lut")
    );

    brdfLutSet = std::make_shared<DescriptorSet>(context, 1);

    brdfLutSet->bindImageToDescriptorSet(
        0,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_SHADER_STAGE_COMPUTE_BIT,
        brdfLut
    );
    brdfLutSet->build();

    brdfLutPipeline->addDescriptorSet(0, brdfLutSet);

    brdfLutPipeline->build();
}

void IblEnvPass::destroyBrdfLutPipeline()
{
    brdfLutSet.reset();
    brdfLutPipeline.reset();
}

void IblEnvPass::createSheenPrefilterResources()
{
    destroySheenPrefilterResources();

    if (!context) throw std::runtime_error("IblEnvPass::createSheenPrefilterResources: context null");

    sheenPrefilterMipLevels = calcMipLevels((uint32_t)sheenPrefilterSize);

    sheenPrefilterCube = std::make_shared<Image>();
    sheenPrefilterCube->format = VK_FORMAT_R16G16B16A16_SFLOAT;
    sheenPrefilterCube->extent = VkExtent2D{ (uint32_t)sheenPrefilterSize, (uint32_t)sheenPrefilterSize };

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = sheenPrefilterCube->format;
    ici.extent = { (uint32_t)sheenPrefilterSize, (uint32_t)sheenPrefilterSize, 1 };
    ici.mipLevels = sheenPrefilterMipLevels;
    ici.arrayLayers = 6;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(
        context->allocator, &ici, &ainfo,
        &sheenPrefilterCube->image, &sheenPrefilterCube->allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("IblEnvPass::createSheenPrefilterResources: vmaCreateImage failed");
    }

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = sheenPrefilterCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format = sheenPrefilterCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = sheenPrefilterMipLevels;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &sheenPrefilterCube->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createSheenPrefilterResources: create 2D_ARRAY view failed");
        }
    }

    sheenPrefilterCubeSampled = std::make_shared<Image>();
    sheenPrefilterCubeSampled->image = sheenPrefilterCube->image;
    sheenPrefilterCubeSampled->format = sheenPrefilterCube->format;
    sheenPrefilterCubeSampled->extent = sheenPrefilterCube->extent;
    sheenPrefilterCubeSampled->allocation = VK_NULL_HANDLE;

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = sheenPrefilterCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        vci.format = sheenPrefilterCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = sheenPrefilterMipLevels;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &sheenPrefilterCubeSampled->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createSheenPrefilterResources: create CUBE view failed");
        }
    }

    sheenPrefilterMipViews.clear();
    sheenPrefilterMipViews.resize(sheenPrefilterMipLevels);

    for (uint32_t mip = 0; mip < sheenPrefilterMipLevels; ++mip) {
        auto v = std::make_shared<Image>();
        v->image = sheenPrefilterCube->image;
        v->format = sheenPrefilterCube->format;
        v->extent = sheenPrefilterCube->extent;
        v->allocation = VK_NULL_HANDLE;

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = sheenPrefilterCube->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format = sheenPrefilterCube->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = mip;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 6;

        if (vkCreateImageView(context->device, &vci, nullptr, &v->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createSheenPrefilterResources: create per-mip 2D_ARRAY view failed");
        }

        sheenPrefilterMipViews[mip] = v;
    }

    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        info.minLod = 0.0f;
        info.maxLod = (float)(sheenPrefilterMipLevels - 1);
        info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(context->device, &info, nullptr, &sheenPrefilterSampler) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createSheenPrefilterResources: create sampler failed");
        }
    }

    sheenPrefilterLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::destroySheenPrefilterResources() 
{
    if (!context) return;

    for (auto& v : sheenPrefilterMipViews) {
        if (v && v->imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(context->device, v->imageView, nullptr);
            v->imageView = VK_NULL_HANDLE;
        }
    }
    sheenPrefilterMipViews.clear();

    if (sheenPrefilterCubeSampled && sheenPrefilterCubeSampled->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, sheenPrefilterCubeSampled->imageView, nullptr);
        sheenPrefilterCubeSampled->imageView = VK_NULL_HANDLE;
    }
    sheenPrefilterCubeSampled.reset();

    if (sheenPrefilterCube && sheenPrefilterCube->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, sheenPrefilterCube->imageView, nullptr);
        sheenPrefilterCube->imageView = VK_NULL_HANDLE;
    }
    if (sheenPrefilterCube && sheenPrefilterCube->image != VK_NULL_HANDLE && sheenPrefilterCube->allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, sheenPrefilterCube->image, sheenPrefilterCube->allocation);
        sheenPrefilterCube->image = VK_NULL_HANDLE;
        sheenPrefilterCube->allocation = VK_NULL_HANDLE;
    }
    sheenPrefilterCube.reset();

    sheenPrefilterMipLevels = 0;
    sheenPrefilterLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::createSheenPrefilterPipeline()
{
    destroySheenPrefilterPipeline();

    if (!envCubeSampled || envCubeSampler == VK_NULL_HANDLE)
        throw std::runtime_error("createSheenPrefilterPipeline: envCube not ready");
    if (sheenPrefilterMipViews.empty() || sheenPrefilterMipLevels == 0)
        throw std::runtime_error("createSheenPrefilterPipeline: prefilter mip views not ready");

    sheenPrefilterPipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "prefilter_sheen_env")
    );

    sheenPrefilterSet = std::make_shared<DescriptorSet>(context, 1);

    sheenPrefilterSet->maxOptions = sheenPrefilterMipLevels;

    sheenPrefilterSet->bindCombinedImageSamplerToDescriptorSet(
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        envCubeSampled,
        envCubeSampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    for (uint32_t mip = 0; mip < sheenPrefilterMipLevels; ++mip) {
        sheenPrefilterSet->bindImageToDescriptorSet(
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_COMPUTE_BIT,
            sheenPrefilterMipViews[mip]
        );
    }

    sheenPrefilterSet->build();

    sheenPrefilterPipeline->addDescriptorSet(0, sheenPrefilterSet);

    sheenPrefilterPipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, 16);

    sheenPrefilterPipeline->build();
}

void IblEnvPass::destroySheenPrefilterPipeline()
{
    sheenPrefilterSet.reset();
    sheenPrefilterPipeline.reset();
}

void IblEnvPass::createSheenLutResources()
{
    destroySheenLutResources();

    if (!context) throw std::runtime_error("IblEnvPass::createSheenfLutResources: context null");

    sheenLut = std::make_shared<Image>();
    sheenLut->format = VK_FORMAT_R16G16_SFLOAT;
    sheenLut->extent = VkExtent2D{ (uint32_t)sheenLutSize, (uint32_t)sheenLutSize };

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = sheenLut->format;
    ici.extent = { (uint32_t)sheenLutSize, (uint32_t)sheenLutSize, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(context->allocator, &ici, &ainfo,
        &sheenLut->image, &sheenLut->allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("IblEnvPass::createSheenLutResources: vmaCreateImage failed");
    }

    {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = sheenLut->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = sheenLut->format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount = 1;

        if (vkCreateImageView(context->device, &vci, nullptr, &sheenLut->imageView) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createSheenLutResources: create view failed");
        }
    }

    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        info.minLod = 0.0f;
        info.maxLod = 0.0f;
        info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        if (vkCreateSampler(context->device, &info, nullptr, &sheenLutSampler) != VK_SUCCESS) {
            throw std::runtime_error("IblEnvPass::createBrdfLutResources: create sampler failed");
        }
    }

    sheenLutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::destroySheenLutResources()
{
    if (!context) return;
    if (sheenLut && sheenLut->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(context->device, sheenLut->imageView, nullptr);
        sheenLut->imageView = VK_NULL_HANDLE;
    }
    if (sheenLut && sheenLut->image != VK_NULL_HANDLE && sheenLut->allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(context->allocator, sheenLut->image, sheenLut->allocation);
        sheenLut->image = VK_NULL_HANDLE;
        sheenLut->allocation = VK_NULL_HANDLE;
    }
    sheenLut.reset();
    sheenLutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void IblEnvPass::createSheenLutPipeline()
{
    destroySheenLutPipeline();
    if (!sheenLut || sheenLut->imageView == VK_NULL_HANDLE) {
        throw std::runtime_error("IblEnvPass::createSheenLutPipeline: sheenLut not ready");
    }
    sheenLutPipeline = std::make_shared<ComputePipeline>(
        context,
        std::make_shared<Shader>(context, "sheen_lut")
    );
    sheenLutSet = std::make_shared<DescriptorSet>(context, 1);
    sheenLutSet->bindImageToDescriptorSet(
        0,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_SHADER_STAGE_COMPUTE_BIT,
        sheenLut
    );
    sheenLutSet->build();
    sheenLutPipeline->addDescriptorSet(0, sheenLutSet);
    sheenLutPipeline->build();
}

void IblEnvPass::destroySheenLutPipeline()
{
    sheenLutSet.reset();
    sheenLutPipeline.reset();
}

void IblEnvPass::runEquirectToCubemapOnce()
{
    if (!convertPipeline || !convertSet) return;

    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    transitionImageLayers(
        cmd,
        envCube->image,
        envCubeLayout,
        VK_IMAGE_LAYOUT_GENERAL,
        0, 6,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    envCubeLayout = VK_IMAGE_LAYOUT_GENERAL;

    convertPipeline->bind(cmd, /*frameIndex*/ 0, /*variant*/ 0);

    uint32_t s = (uint32_t)cubeSize;
    vkCmdPushConstants(
        cmd,
        convertPipeline->getPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(uint32_t),
        &s
    );

    uint32_t gx = ((uint32_t)cubeSize + 15u) / 16u;
    uint32_t gy = ((uint32_t)cubeSize + 15u) / 16u;
    vkCmdDispatch(cmd, gx, gy, 6);

    transitionImageLayers(
        cmd,
        envCube->image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        0, 6,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    envCubeLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::Type::GRAPHICS);
}

void IblEnvPass::runIrradianceConvolutionOnce()
{
    if (!irradiancePipeline || !irradianceSet) return;

    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    transitionImageLayers(
        cmd,
        irrCube->image,
        irrCubeLayout,
        VK_IMAGE_LAYOUT_GENERAL,
        0, 6,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    irrCubeLayout = VK_IMAGE_LAYOUT_GENERAL;

    irradiancePipeline->bind(cmd, 0, 0);

    uint32_t s = (uint32_t)irradianceSize;
    vkCmdPushConstants(
        cmd,
        irradiancePipeline->getPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(uint32_t),
        &s
    );

    uint32_t gx = ((uint32_t)irradianceSize + 7u) / 8u;
    uint32_t gy = ((uint32_t)irradianceSize + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 6);

    transitionImageLayers(
        cmd,
        irrCube->image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        0, 6,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
    );
    irrCubeLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::Type::GRAPHICS);
}

void IblEnvPass::runPrefilterOnce()
{
    if (!prefilterPipeline || !prefilterSet) return;
    if (!prefilterCube || prefilterMipLevels == 0) return;

    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    transitionImageSubresources(
        cmd,
        prefilterCube->image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        /*baseMip*/ 0, (uint32_t)prefilterMipLevels,
        /*baseLayer*/ 0, /*layerCount*/ 6,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    prefilterLayout = VK_IMAGE_LAYOUT_GENERAL;

    for (uint32_t mip = 0; mip < (uint32_t)prefilterMipLevels; ++mip) {

        uint32_t outSize = (uint32_t)prefilterSize >> mip;
        if (outSize == 0) outSize = 1;

        float roughness = (prefilterMipLevels <= 1)
            ? 0.0f
            : float(mip) / float(prefilterMipLevels - 1);

        prefilterPipeline->bind(cmd, /*frameIndex*/ 0, /*variant*/ mip);

        struct PushConst {
            uint32_t outSize;
            float    roughness;
            uint32_t mipLevel;
            uint32_t mipCount;
        } pc{ outSize, roughness, mip, (uint32_t)prefilterMipLevels };

        vkCmdPushConstants(
            cmd,
            prefilterPipeline->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConst),
            &pc
        );

        uint32_t gx = (outSize + 7u) / 8u;
        uint32_t gy = (outSize + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 6);
    }

    transitionImageSubresources(
        cmd,
        prefilterCube->image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        /*baseMip*/ 0, (uint32_t)prefilterMipLevels,
        /*baseLayer*/ 0, /*layerCount*/ 6,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    prefilterLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::Type::GRAPHICS);
}

void IblEnvPass::runBrdfLutOnce()
{
    if (!brdfLutPipeline || !brdfLutSet) return;
    if (!brdfLut || brdfLut->image == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    transitionImage(
        cmd,
        brdfLut->image,
        brdfLutLayout,
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    brdfLutLayout = VK_IMAGE_LAYOUT_GENERAL;

    brdfLutPipeline->bind(cmd, 0, 0);

    uint32_t size = (uint32_t)brdfLutSize;
    uint32_t gx = (size + 15u) / 16u;
    uint32_t gy = (size + 15u) / 16u;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(
        cmd,
        brdfLut->image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
    );
    brdfLutLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::Type::GRAPHICS);
}

void IblEnvPass::runSheenPrefilterOnce()
{
    if (!sheenPrefilterPipeline || !sheenPrefilterSet) return;
    if (!sheenPrefilterCube || sheenPrefilterMipLevels == 0) return;

    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    transitionImageSubresources(
        cmd,
        sheenPrefilterCube->image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        /*baseMip*/ 0, (uint32_t)sheenPrefilterMipLevels,
        /*baseLayer*/ 0, /*layerCount*/ 6,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    sheenPrefilterLayout = VK_IMAGE_LAYOUT_GENERAL;

    for (uint32_t mip = 0; mip < (uint32_t)sheenPrefilterMipLevels; ++mip) {

        uint32_t outSize = (uint32_t)sheenPrefilterSize >> mip;
        if (outSize == 0) outSize = 1;

        float roughness = (sheenPrefilterMipLevels <= 1)
            ? 0.0f
            : float(mip) / float(sheenPrefilterMipLevels - 1);

        sheenPrefilterPipeline->bind(cmd, /*frameIndex*/ 0, /*variant*/ mip);

        struct PushConst {
            uint32_t outSize;
            float    roughness;
            uint32_t mipLevel;
            uint32_t mipCount;
        } pc{ outSize, roughness, mip, (uint32_t)sheenPrefilterMipLevels };

        vkCmdPushConstants(
            cmd,
            sheenPrefilterPipeline->getPipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConst),
            &pc
        );

        uint32_t gx = (outSize + 7u) / 8u;
        uint32_t gy = (outSize + 7u) / 8u;
        vkCmdDispatch(cmd, gx, gy, 6);
    }

    transitionImageSubresources(
        cmd,
        sheenPrefilterCube->image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        /*baseMip*/ 0, (uint32_t)sheenPrefilterMipLevels,
        /*baseLayer*/ 0, /*layerCount*/ 6,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    sheenPrefilterLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::Type::GRAPHICS);
}

void IblEnvPass::runSheenLutOnce()
{
    if (!sheenLutPipeline || !sheenLutSet) return;
    if (!sheenLut || sheenLut->image == VK_NULL_HANDLE) return;
    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();
    transitionImage(
        cmd,
        sheenLut->image,
        sheenLutLayout,
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    sheenLutLayout = VK_IMAGE_LAYOUT_GENERAL;
    sheenLutPipeline->bind(cmd, 0, 0);
    uint32_t size = (uint32_t)sheenLutSize;
    uint32_t gx = (size + 15u) / 16u;
    uint32_t gy = (size + 15u) / 16u;
    vkCmdDispatch(cmd, gx, gy, 1);
    transitionImage(
        cmd,
        sheenLut->image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
    );
    sheenLutLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::Type::GRAPHICS);
}

void IblEnvPass::loadHDRToTexture2D(const std::string& path)
{
    int w = 0, h = 0, comp = 0;
    float* pixels = stbi_loadf(path.c_str(), &w, &h, &comp, 3);
    if (!pixels) {
        throw std::runtime_error(std::string("IblEnvPass: stbi_loadf failed: ") + stbi_failure_reason());
    }

    spdlog::info("IblEnvPass: loaded HDR {} ({}x{}, comp={})", path, w, h, comp);

    std::vector<float> rgba;
    rgba.resize(size_t(w) * size_t(h) * 4);
    for (int i = 0; i < w * h; ++i) {
        rgba[i * 4 + 0] = pixels[i * 3 + 0];
        rgba[i * 4 + 1] = pixels[i * 3 + 1];
        rgba[i * 4 + 2] = pixels[i * 3 + 2];
        rgba[i * 4 + 3] = 1.0f;
    }
    stbi_image_free(pixels);

    const VkDeviceSize byteSize = VkDeviceSize(rgba.size() * sizeof(float));

    // staging
    auto staging = Buffer::staging(context, (unsigned long)byteSize);
    std::memcpy(staging->allocation_info.pMappedData, rgba.data(), (size_t)byteSize);

    // image create
    envEquirect = std::make_shared<Image>();
    envEquirect->format = VK_FORMAT_R32G32B32A32_SFLOAT;
    envEquirect->extent = VkExtent2D{ (uint32_t)w, (uint32_t)h };

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = envEquirect->format;
    ici.extent = { (uint32_t)w, (uint32_t)h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ainfo{};
    ainfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(context->allocator, &ici, &ainfo,
        &envEquirect->image, &envEquirect->allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("IblEnvPass: vmaCreateImage failed");
    }

    // view
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = envEquirect->image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = envEquirect->format;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel = 0;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount = 1;

    if (vkCreateImageView(context->device, &vci, nullptr, &envEquirect->imageView) != VK_SUCCESS) {
        throw std::runtime_error("IblEnvPass: vkCreateImageView failed");
    }

    // copy staging -> image
    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();
    transitionImage(
        cmd,
        envEquirect->image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };

    vkCmdCopyBufferToImage(
        cmd,
        staging->buffer,
        envEquirect->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    transitionImage(
        cmd,
        envEquirect->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::Type::GRAPHICS);
}
