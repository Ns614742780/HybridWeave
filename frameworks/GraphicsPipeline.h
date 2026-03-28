#pragma once

#include "Pipeline.h"
#include "Shader.h"

#include <vector>
#include <cstdint>

class GraphicsPipeline : public Pipeline
{
public:
    struct VertexInputDesc {
        VkVertexInputBindingDescription binding{};
        std::vector<VkVertexInputAttributeDescription> attributes;
    };

    explicit GraphicsPipeline(const std::shared_ptr<VulkanContext>& context);
    ~GraphicsPipeline() override = default;

    void setRenderPass(VkRenderPass rp, uint32_t subpass = 0);
    void setShaders(const std::shared_ptr<Shader>& vert, const std::shared_ptr<Shader>& frag);

    void setVertexInput(const VertexInputDesc& vi);
    void setExtent(VkExtent2D extent);
    void setCullMode(VkCullModeFlags cull);
    void setFrontFace(VkFrontFace ff);
    void setDepthTest(bool enabled);
    void setColorAttachmentCount(uint32_t count);
    void setFlippedViewport(bool flipped);
    void GraphicsPipeline::setColorBlendAttachments(const std::vector<VkPipelineColorBlendAttachmentState>& states);
    void setFragmentShadingRateState(
        VkExtent2D fragmentSize,
        VkFragmentShadingRateCombinerOpKHR combiner0,
        VkFragmentShadingRateCombinerOpKHR combiner1);

    void enableDynamicViewportScissor(bool enable);
    void enableDynamicFragmentShadingRate(bool enable);
    void enableFragmentShadingRateState(bool enable);

    void build() override;
    void bindWithSet(
        VkCommandBuffer cmd,
        uint8_t currentFrame,
        uint8_t optionIndex,
        std::shared_ptr<DescriptorSet> set);
    void GraphicsPipeline::bindWithSets(
        VkCommandBuffer cmd,
        uint8_t frameIndex,
        uint8_t optionIndex,
        const std::vector<std::shared_ptr<DescriptorSet>>& sets);

    // IMPORTANT: graphics bind point
    void bind(VkCommandBuffer commandBuffer,
        uint8_t currentFrame,
        DescriptorOption option) override;

private:
    VkRenderPass renderPass = VK_NULL_HANDLE;
    uint32_t subpass = 0;

    std::shared_ptr<Shader> vertShader;
    std::shared_ptr<Shader> fragShader;

    VertexInputDesc vertexInput{};

    VkExtent2D extent{ 0, 0 };

    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	uint32_t colorAttachmentCount = 1;
    bool depthTest = false;
	bool flippedViewport = false;
    bool dynamicViewportScissor = false;
    bool dynamicFragmentShadingRate = false;
	bool fragmentShadingRateStateEnabled = false;

    std::vector<VkPipelineColorBlendAttachmentState> customColorBlendAttachments;

    VkExtent2D fsrFragmentSize{ 1, 1 };
    VkFragmentShadingRateCombinerOpKHR fsrCombinerOps[2] = {
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR
    };
    
};
