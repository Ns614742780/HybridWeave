#pragma once

#include "IRenderPass.h"
#include "IColorProvider.h"
#include "RenderGlobalResources.h"
#include "GraphicsPipeline.h"
#include "DescriptorSet.h"
#include "Buffer.h"

#include <memory>
#include <vector>

// forward declare
class IGBufferProvider;
class IblEnvPass;
class DescriptorSet;
struct Image;

class LightingPass : public IRenderPass, public IColorProvider
{
public:
    // debugView:
    struct PushConst {
        int debugView;
        int _pad0;
        int _pad1;
        int _pad2;
    };
    // 0=albedo, 1=normal, 2=worldPos, 3=emissiveAO, 4=depth
    LightingPass(const RenderGlobalResources& r,
        IGBufferProvider* gbufferSource,
		IblEnvPass* iblEnv,
        int debugView = 0);

    ~LightingPass() override;

    void initialize() override;
    void onSwapchainResized() override;
    void record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex) override;
    void update(float dt) override;

    // IColorProvider
    const std::shared_ptr<Image>& getColorImage(uint32_t imageIndex) const override { return lightingColor[imageIndex]; }
    VkImageLayout getColorLayout(uint32_t imageIndex) const override { return lightingColorLayouts[imageIndex]; }
    VkFormat getColorFormat() const override { return lightingFormat; }
    VkExtent2D getColorExtent() const override { return swapchain->swapchainExtent; }

    void setDebugView(int v) { debugView = v; }

private:
    void createRenderPassAndFramebuffers();
    void destroyRenderPassAndFramebuffers();

    void createPipelineAndDescriptors();
    void destroyPipelineAndDescriptors();

    void createFullscreenTriangleVB();
    void destroyFullscreenTriangleVB();

    void createOffscreenColor();
    void destroyOffscreenColor();

    void createSampler();
    void destroySampler();

private:
    RenderGlobalResources global;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<Swapchain> swapchain;
	std::shared_ptr<QueryManager> queryManager;

    // source of gbuffer images (owned by GltfRenderPass)
    IGBufferProvider* gbufferPass = nullptr;
	IblEnvPass* iblEnvPass = nullptr;

    int debugView = 0;

    // RenderPass/FB for presenting to swapchain
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    // Track swapchain image layouts like your other passes
    VkFormat lightingFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    std::vector<std::shared_ptr<Image>> lightingColor;
    std::vector<VkImageLayout> lightingColorLayouts;

    // Pipeline & descriptor sets
    std::shared_ptr<GraphicsPipeline> pipeline;

    std::shared_ptr<DescriptorSet> gbufferSet;
    std::shared_ptr<DescriptorSet> lightingSet;
    std::shared_ptr<DescriptorSet> uniformSet;

    VkSampler gbufferSampler = VK_NULL_HANDLE;
    VkSampler envSampler = VK_NULL_HANDLE;
	VkSampler materialSampler = VK_NULL_HANDLE;

    // Fullscreen triangle vertex buffer
    std::shared_ptr<Buffer> fsTriangleVB;
};
