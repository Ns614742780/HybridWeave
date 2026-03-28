#pragma once

#include "IRenderPass.h"
#include "RenderGlobalResources.h"
#include "RendererConfiguration.h"
#include "ComputePipeline.h"
#include "DescriptorSet.h"
#include "Image.h"
#include "Buffer.h"

#include <memory>
#include <string>

class IblEnvPass : public IRenderPass {
public:
    IblEnvPass(const RenderGlobalResources& r, std::string hdrPath, float intensity = 1.0f);
    ~IblEnvPass() override;

    void initialize() override;
    void onSwapchainResized() override {}
    void record(VkCommandBuffer, uint32_t, uint32_t) override {}
    void update(float) override {}

    std::shared_ptr<Image> getEnvEquirectImage() const { return envEquirect; }
    VkSampler              getEnvSampler() const { return envSampler; }
    float                  getIntensity() const { return envIntensity; }

    std::shared_ptr<Image> getEnvCubemapImage() const { return envCubeSampled ? envCubeSampled : envCube; }
    VkSampler              getEnvCubemapSampler() const { return envCubeSampler; }

    std::shared_ptr<Image> getIrradianceCubemapImage() const { return irrCubeSampled; }
    VkSampler getIrradianceCubemapSampler() const { return irrCubeSampler; }

	std::shared_ptr<Image> getPrefilterCubemapImage() const { return prefilterCubeSampled; }
	VkSampler getPrefilterCubemapSampler() const { return prefilterSampler; }

    std::shared_ptr<Image> getBrdfLutImage() const { return brdfLut; }
	VkSampler getBrdfLutSampler() const { return brdfLutSampler; }

    uint32_t getPrefilterMipLevels() const { return prefilterMipLevels; }

    std::shared_ptr<Image> getSheenPrefilterCubemapImage() const { return sheenPrefilterCubeSampled; }
    VkSampler getSheenPrefilterCubemapSampler() const { return sheenPrefilterSampler; }

    std::shared_ptr<Image> getSheenLutImage() const { return sheenLut; }
    VkSampler getSheenLutSampler() const { return sheenLutSampler; }
    uint32_t getSheenPrefilterMipLevels() const { return sheenPrefilterMipLevels; }

private:
    void createSampler();
    void destroySampler();

    void createCubemapResources();
    void destroyCubemapResources();

    void createConvertPipeline();   
    void destroyConvertPipeline();

    void createIrradianceResources();
    void destroyIrradianceResources();

    void createIrradiancePipeline();
    void destroyIrradiancePipeline();

	void createPrefilterResources();
	void destroyPrefilterResources();

	void createPrefilterPipeline();
	void destroyPrefilterPipeline();

	void createBrdfLutResources();
	void destroyBrdfLutResources();

	void createBrdfLutPipeline();
	void destroyBrdfLutPipeline();

    void createSheenPrefilterResources();
    void destroySheenPrefilterResources();

    void createSheenPrefilterPipeline();
    void destroySheenPrefilterPipeline();

    void createSheenLutResources();
    void destroySheenLutResources();

    void createSheenLutPipeline();
    void destroySheenLutPipeline();

    void runIrradianceConvolutionOnce();
    void runEquirectToCubemapOnce();
	void runPrefilterOnce();
	void runBrdfLutOnce();
    void runSheenPrefilterOnce();
    void runSheenLutOnce();

    void loadHDRToTexture2D(const std::string& path);

private:
    RenderGlobalResources global{};
    std::shared_ptr<VulkanContext> context;

    std::string hdrPath;
    float envIntensity = 1.0f;

    std::shared_ptr<Image> envEquirect;
    VkSampler envSampler = VK_NULL_HANDLE;

    // Cubemap GPU resource (compute writes here)
    std::shared_ptr<Image> envCube = nullptr;   // image: 2D array, 6 layers (OWNING)
    std::shared_ptr<Image> envCubeSampled = nullptr;
    VkImageLayout envCubeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // Sampling wrapper (same VkImage, cube view, NON-OWNING allocation)
    VkSampler envCubeSampler = VK_NULL_HANDLE;

    // ---- irradiance cubemap ----
    std::shared_ptr<Image> irrCube = nullptr;            // 2D_ARRAY view (storage)
    std::shared_ptr<Image> irrCubeSampled = nullptr;     // CUBE view (sampling wrapper)
    VkImageLayout irrCubeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler irrCubeSampler = VK_NULL_HANDLE;

    std::shared_ptr<Image> prefilterCube = nullptr;
    std::shared_ptr<Image> prefilterCubeSampled = nullptr;
    VkImageLayout prefilterLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler prefilterSampler = VK_NULL_HANDLE;

    std::vector<std::shared_ptr<Image>> prefilterMipViews;

    std::shared_ptr<Image> brdfLut;
    VkImageLayout brdfLutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler brdfLutSampler = VK_NULL_HANDLE;

    // ---- sheen prefilter cubemap ----
    std::shared_ptr<Image> sheenPrefilterCube = nullptr;         // 2D_ARRAY storage
    std::shared_ptr<Image> sheenPrefilterCubeSampled = nullptr;  // CUBE sampling view
    VkImageLayout sheenPrefilterLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler sheenPrefilterSampler = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<Image>> sheenPrefilterMipViews;

    // ---- sheen LUT (DFG) ----
    std::shared_ptr<Image> sheenLut = nullptr;
    VkImageLayout sheenLutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler sheenLutSampler = VK_NULL_HANDLE;

    // compute: one set, one pipeline (align to your existing members)
    std::shared_ptr<DescriptorSet>   convertSet;
    std::shared_ptr<ComputePipeline> convertPipeline;

    std::shared_ptr<DescriptorSet>   irradianceSet;
    std::shared_ptr<ComputePipeline> irradiancePipeline;

    std::shared_ptr<DescriptorSet> prefilterSet;
    std::shared_ptr<ComputePipeline> prefilterPipeline;

    std::shared_ptr<DescriptorSet> brdfLutSet;
    std::shared_ptr<ComputePipeline> brdfLutPipeline;

    std::shared_ptr<DescriptorSet> sheenPrefilterSet;
    std::shared_ptr<ComputePipeline> sheenPrefilterPipeline;

    std::shared_ptr<DescriptorSet> sheenLutSet;
    std::shared_ptr<ComputePipeline> sheenLutPipeline;

    int cubeSize = 512;
    int irradianceSize = 32;
    int prefilterSize = 256;
    int brdfLutSize = 256;
    int prefilterMipLevels = 0;
    int sheenPrefilterSize = 256;
    int sheenLutSize = 256;
    int sheenPrefilterMipLevels = 0;
};
