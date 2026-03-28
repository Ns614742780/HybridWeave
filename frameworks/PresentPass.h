#pragma once

#include "IRenderPass.h"
#include "IColorProvider.h"
#include "RenderGlobalResources.h"
#include "GraphicsPipeline.h"
#include "ComputePipeline.h"
#include "Buffer.h"

#include <memory>
#include <vector>

class DescriptorSet;
class IGBufferProvider;
class IDepthProvider;

class PresentPass : public IRenderPass
{
public:
    enum class Mode : int {
        PresentA = 0,  // show srcA
        PresentB = 1,  // show srcB
        Mix = 2,  // composite/mix
    };
    struct Params {
        Mode  mode = Mode::PresentA;
		bool  openMixEnhance = false;
        int   mixOp = 0;
        float mixFactor = 0.5f;
		int   styleLock = 0;
        int   enableBlurMap = 1;        // 0/1
        float blurMapStrength = 1.0f;   // global strength scale
        int   stylePreset = 1;          // 0=default, 1=outdoor_veg
    };
    struct AutoMatchParamsGPU {
        float gain;
        float wbR;
        float wbG;
        float wbB;
        glm::vec4 blurStats;
    };
    struct AutoMatchPC2
    {
        int   statsW;
        int   statsH;
        int   fullW;
        int   fullH;

        int   ringRadius;
        int   ringSamples;
        int   stride;
        int   styleLock;

        float coverageTh;
        float emaAlpha;
        float maxDeltaGain;
        float maxDeltaWb;

        float minGain;
        float maxGain;
        float reserved0;
        float reserved1;
    };

    explicit PresentPass(const RenderGlobalResources& r, 
        IColorProvider* srcA, IColorProvider* srcB = nullptr, 
        Params p = {});
    ~PresentPass() override;

    void initialize() override;
    void onSwapchainResized() override;
    void record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex) override;
    void update(float dt) override;

    void setParams(Params p) { params = p; }
    void setGBufferProvider(IGBufferProvider* p) { gbufferProvider = p; }

private:
    void createStage2StatsImages(uint32_t frames);
    void createStage2DownsamplePipeline(uint32_t frames);
    void createBlurMapImages(uint32_t frames);
    void rebuildPresentDescriptorsAndPipelineOnly();

    void createRenderPassAndFramebuffers();
    void destroyRenderPassAndFramebuffers();

    void createPipelineAndDescriptors();
    void destroyPipelineAndDescriptors();

    void createFullscreenTriangleVB();
    void destroyFullscreenTriangleVB();

    void createSampler();
    void destroySampler();

    void destroySwapchainDependentResources();
    void destroyResources();

    struct PushConst {
        int   presentMode;   // 0/1/2
        int   mixOp;         // 0/1
        float mixFactor;     // 0..1
		float alphaPow;      // >=1 strongth for edge solidness suggested 2-4

		float featherRange;  // only improve alpha when depth is saperate indeed 
		float depthEps;      // depth compare epsilon suggested 1e-6 - 1e-4
		int   useMinDepthA;  // reduce gltf outline leakage
		int   styleLock;     // 0 = normal, 1 = lock to style to stay stable
    };
    struct DownsamplePC {
        int fullW;
        int fullH;
        int statsW;
        int statsH;
    };
    struct BlurMapPC {
        int width;
        int height;

        float emaAlpha;         // 0..1
        float covTh;            // low threshold
        float covStrongTh;      // strong threshold
        float baseSoft;         // baseline softening even when coverage low

        float maxSoft;          // clamp 0..1
        float depthTolBase;     // base tolerance
        float depthTolK;        // scale with depth
        float reserved0;

        int styleLock;
        int preset;
        float reserved1;
        float reserved2;
    };
private:
    RenderGlobalResources global;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<Swapchain> swapchain;

    IColorProvider* srcA = nullptr;
    IColorProvider* srcB = nullptr;
    IGBufferProvider* gbufferProvider = nullptr;
    IDepthProvider* gsDepthProvider = nullptr;
    Params params{};

    bool lutInited = false;
    uint32_t statsW = 320;
    uint32_t statsH = 180;

    std::vector<std::shared_ptr<Image>> statsColorA;  // per-frame
    std::vector<std::shared_ptr<Image>> statsColorB;
    std::vector<VkImageLayout> statsColorALayouts;
    std::vector<VkImageLayout> statsColorBLayouts;

    std::shared_ptr<ComputePipeline> downsamplePipeline;
    std::shared_ptr<DescriptorSet>   downsampleSet;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;
    std::vector<VkImageLayout> swapImageLayouts;

    std::shared_ptr<ComputePipeline> autoMatchPipeline;
    std::shared_ptr<DescriptorSet>   autoMatchSet;
    std::shared_ptr<DescriptorSet> enhanceSet;
    std::vector<std::shared_ptr<Buffer>> autoMatchParams;

    std::vector<uint32_t> autoMatchPing;
    std::vector<std::shared_ptr<Image>> autoMatchLutPrev;
    std::vector<std::shared_ptr<Image>> autoMatchLutNext;
    std::vector<VkImageLayout> autoMatchLutPrevLayouts;
    std::vector<VkImageLayout> autoMatchLutNextLayouts;

    std::shared_ptr<ComputePipeline> lutInitPipeline;
    std::shared_ptr<DescriptorSet>   lutInitSet;

    std::vector<std::shared_ptr<Image>> blurMapPrev;
    std::vector<std::shared_ptr<Image>> blurMapNext;
    std::vector<VkImageLayout> blurMapPrevLayouts;
    std::vector<VkImageLayout> blurMapNextLayouts;

    std::shared_ptr<ComputePipeline> blurMapPipeline;
    std::shared_ptr<DescriptorSet>   blurMapSet;       // compute
    std::shared_ptr<DescriptorSet>   blurMapFragSet;   // fragment sampled blurMapNext

    VkSampler nearestSampler = VK_NULL_HANDLE;

    std::shared_ptr<GraphicsPipeline> pipeline;
    std::shared_ptr<DescriptorSet> srcSet;

    VkSampler srcSampler = VK_NULL_HANDLE;
    std::shared_ptr<Buffer> fsTriangleVB;
};
