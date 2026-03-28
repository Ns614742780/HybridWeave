#pragma once

#include "IRenderPass.h"
#include "RenderGlobalResources.h"
#include "GltfLoader.h"
#include "IGBufferProvider.h"
#include "GraphicsPipeline.h"

#include <memory>
#include <vector>
#include <string>

class GltfRenderPass : public IRenderPass, public IGBufferProvider
{
public:

    explicit GltfRenderPass(const RenderGlobalResources& r, const std::string& scenePath);
    ~GltfRenderPass() override;

    void initialize() override;
    void onSwapchainResized() override;
    void record(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t frameIndex) override;
	void update(float dt) override;

private:
    void createRenderPassAndFramebuffers();
    void destroyRenderPassAndFramebuffers();

    void createPipelineAndDescriptors();
    void destroyPipelineAndDescriptors();

    void createDummyWhiteTexture();

    void updateUniforms();

    uint32_t pickTextureOptionForMesh(const GltfMeshGPU& m) const;

private:
    RenderGlobalResources global;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<Swapchain> swapchain;
	std::shared_ptr<Buffer> uniformBuffer;
    std::shared_ptr<QueryManager>  queryManager;
    Camera* camera = nullptr;

    std::string scenePath;

    std::unique_ptr<GltfLoaderVulkan> loader;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    std::shared_ptr<GraphicsPipeline> pipeline;
    std::shared_ptr<DescriptorSet>    mainSet;

    std::vector<std::shared_ptr<Image>> textureImageWrappers;

    std::vector<IGBufferProvider::GBufferImages> gbufferImages;

	VkFormat albedoRoughFormat = VK_FORMAT_R8G8B8A8_UNORM;
	VkFormat normalMetalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
	VkFormat worldPosFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
	VkFormat emissiveAOFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat sheenFormat = VK_FORMAT_R8G8B8A8_UNORM;
	VkFormat materialFormat = VK_FORMAT_R32_UINT;
    VkFormat drawIdFormat = VK_FORMAT_R32_UINT;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    std::shared_ptr<Image> dummyWhiteImage;
    VkSampler              dummySampler = VK_NULL_HANDLE;

    VkImage        dummyImage = VK_NULL_HANDLE;
    VmaAllocation  dummyImageAlloc = VK_NULL_HANDLE;

    // fallback option
    uint32_t fallbackTextureOption = 0;
	uint32_t gbufferMipLevels = 1;

    bool initialized = false;

    struct PushConst {
        glm::mat4 model;
        glm::vec4 baseColorFactor;

        float metallicFactor;
        float roughnessFactor;

        uint32_t  baseColorTex;
        uint32_t  mrTex;  

        uint32_t materialFlags;
        uint32_t drawId;
        uint32_t _pad2;   
        uint32_t _pad3;   

        glm::vec4 sheenColorRoughFactor; // xyz = sheenColorFactor, w = sheenRoughnessFactor
        uint32_t sheenColorTex;          // srgb
        uint32_t sheenRoughTex;          // linear (R)
        uint32_t _pad4;
    };

public:
    const std::vector<IGBufferProvider::GBufferImages>& getGBufferImages() const { return gbufferImages; }
	const uint32_t getGBufferMipLevels() const { return gbufferMipLevels; }

};
