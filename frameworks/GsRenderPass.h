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

#include "RenderGlobalResources.h"
#include "IRenderPass.h"
#include "IColorProvider.h" 
#include "IDepthProvider.h"

class VulkanContext;
class Swapchain;
class Buffer;
class QueryManager;
class Camera;
class GSScene;
class ComputePipeline;
class DescriptorSet;
class IGBufferProvider;
struct Image;

class GsRenderPass : public IRenderPass, public IColorProvider, public IDepthProvider{
public:

    struct VertexAttributeBuffer {
        glm::vec4 conic_opacity;
        glm::vec4 color_radii;
        glm::uvec4 aabb;
        glm::vec2 uv;
        float     depth;
		float     depth01;
    };

    struct RadixSortPushConstants {
        uint32_t g_num_elements;
        uint32_t g_shift;
        uint32_t g_num_workgroups;
        uint32_t g_num_blocks_per_workgroup;
    };

    struct MixPC {
        uint32_t width;
        uint32_t height;
        float projA;
        float projB;
        float projC;
        float farDepth;
    };

    explicit GsRenderPass(
        const RenderGlobalResources& r,
        const std::string& scenePath);
    ~GsRenderPass() override;
    
    void initialize() override;
    void update(float dt) override;
    void record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex) override;
    void onSwapchainResized() override;

	// IColorProvider
    const std::shared_ptr<Image>& getColorImage(uint32_t imageIndex) const override {
        return gsColor.at(imageIndex);
    }
    VkImageLayout getColorLayout(uint32_t imageIndex) const override {
        return gsColorLayouts.at(imageIndex);
    }
	// IDepthProvider
    std::shared_ptr<Image> getDepthImage(uint32_t imageIndex) const
    {
        if (imageIndex >= gsDepth.size()) return nullptr;
        return gsDepth[imageIndex];
    }
    VkFormat getColorFormat() const override { return gsColorFormat; }
    VkExtent2D getColorExtent() const override { return gsExtent; }

    void setOccluderDepthProvider(IGBufferProvider* p) { occluder = p; }
private:

    void loadScene();

	void createBuffers();

    void createPreprocessPipeline();
    void createPrefixSumPipeline();
    void createRadixSortPipeline();
    void createPreprocessSortPipeline();
    void createTileBoundaryPipeline();
    void createRenderPipeline();

    void createRenderPipelineMix();
    void createMeshDepthSampler();
    void destroyMeshDepthSampler();

    void createGsOffscreenImages();
    void destroyGsOffscreenImages();

    void rebuildForResolutionChange();
    void ensureSortCapacity(uint32_t requiredInstances);

    void dispatchPreprocess(VkCommandBuffer cmd, uint32_t frameIndex);
    void dispatchPrefixSum(VkCommandBuffer cmd, uint32_t frameIndex);
    void dispatchRadixSort(VkCommandBuffer cmd, uint32_t frameIndex);
    void dispatchPreprocessSort(VkCommandBuffer cmd, uint32_t frameIndex);
    void dispatchTileBoundary(VkCommandBuffer cmd, uint32_t frameIndex);
    void dispatchRender(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex);

private:
    RenderGlobalResources global;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<Swapchain>     swapchain;
    std::shared_ptr<Buffer>        uniformBuffer;
    std::shared_ptr<QueryManager>  queryManager;
    Camera* camera = nullptr;

    std::shared_ptr<GSScene>       scene;

    std::vector<VkImageLayout> swapImageLayouts;

    std::vector<std::shared_ptr<Image>> gsColor;
    std::vector<VkImageLayout> gsColorLayouts;

    std::vector<std::shared_ptr<Image>> gsDepth;          // R32_SFLOAT, depth01
    std::vector<VkImageLayout>          gsDepthLayouts;

    IGBufferProvider* occluder = nullptr;
    VkSampler meshDepthSampler = VK_NULL_HANDLE;

    VkFormat  gsColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
	VkFormat  gsDepthFormat = VK_FORMAT_R32_SFLOAT;
    VkExtent2D gsExtent{ 0, 0 };
    bool hasTotalSumReadback = false;

    uint32_t gaussianCount = 0;
    uint32_t cachedNumInstances = 0;

    unsigned int sortBufferSizeMultiplier = 1;
    uint32_t numRadixSortBlocksPerWorkgroup = 32;

    std::shared_ptr<Buffer> vertexAttributeBuffer;
    std::shared_ptr<Buffer> tileOverlapBuffer;
    std::shared_ptr<Buffer> prefixSumPingBuffer;
    std::shared_ptr<Buffer> prefixSumPongBuffer;
    std::shared_ptr<Buffer> sortKBufferEven;
    std::shared_ptr<Buffer> sortKBufferOdd;
    std::shared_ptr<Buffer> sortHistBuffer;
    std::shared_ptr<Buffer> totalSumBufferHost;
    std::shared_ptr<Buffer> tileBoundaryBuffer;
    std::shared_ptr<Buffer> sortVBufferEven;
    std::shared_ptr<Buffer> sortVBufferOdd;

    std::shared_ptr<ComputePipeline> preprocessPipeline;

    std::shared_ptr<ComputePipeline> prefixSumPipeline;

    std::shared_ptr<ComputePipeline> preprocessSortPipeline;

    std::shared_ptr<ComputePipeline> sortHistPipeline;
    std::shared_ptr<ComputePipeline> sortPipeline;

    std::shared_ptr<ComputePipeline> tileBoundaryPipeline;

    std::shared_ptr<ComputePipeline> renderPipeline;
    std::shared_ptr<ComputePipeline> renderPipelineMix;

    std::shared_ptr<DescriptorSet> preprocessInputSet;
    std::shared_ptr<DescriptorSet> preprocessOutputSet;

    std::shared_ptr<DescriptorSet> prefixSumSet;

    std::shared_ptr<DescriptorSet> radixHistSet;
	std::shared_ptr<DescriptorSet> radixSortSet;

    std::shared_ptr<DescriptorSet> preprocessSortSet;

    std::shared_ptr<DescriptorSet> tileBoundarySet;

    std::shared_ptr<DescriptorSet> renderInputSet;
    std::shared_ptr<DescriptorSet> renderOutputSet;

    std::shared_ptr<DescriptorSet> renderDepthSet;
};
