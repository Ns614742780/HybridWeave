#include "ComputePipeline.h"
#include <stdexcept>

ComputePipeline::ComputePipeline(const std::shared_ptr<VulkanContext>& context,
    std::shared_ptr<Shader> shader)
    : Pipeline(context), shader(std::move(shader))
{
    if (!this->shader) {
        throw std::runtime_error("ComputePipeline: shader is null");
    }
    this->shader->load();
}

void ComputePipeline::build()
{
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context->device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }

    buildPipelineLayout();

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader->shader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkResult r = vkCreateComputePipelines(
        context->device,
        VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &pipeline
    );

    if (r != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
}

void ComputePipeline::bindWithSet(
    VkCommandBuffer cmd,
    uint8_t frameIndex,
    uint32_t optionIndex,
    std::shared_ptr<DescriptorSet> set)
{
    if (!pipeline || !pipelineLayout)
        throw std::runtime_error("ComputePipeline::bindWithSet: pipeline not built");

    vkCmdBindPipeline(
        cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline
    );

    VkDescriptorSet vkSet =
        set->getDescriptorSet(frameIndex, optionIndex);

    if (vkSet == VK_NULL_HANDLE)
        throw std::runtime_error("ComputePipeline::bindWithSet: VkDescriptorSet is null");

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0,                      // firstSet
        1, &vkSet,              // descriptorSets
        0, nullptr              // dynamic offsets
    );
}

void ComputePipeline::bindWithSets(
    VkCommandBuffer cmd,
    uint8_t frameIndex,
    uint32_t optionIndex,
    const std::vector<std::shared_ptr<DescriptorSet>>& sets)
{
    if (!pipeline || !pipelineLayout)
        throw std::runtime_error("ComputePipeline::bindWithSets: pipeline not built");

    vkCmdBindPipeline(
        cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline
    );

    std::vector<VkDescriptorSet> vkSets;
    vkSets.reserve(sets.size());

    for (size_t i = 0; i < sets.size(); ++i)
    {
        if (!sets[i])
            throw std::runtime_error("ComputePipeline::bindWithSets: null DescriptorSet");

        VkDescriptorSet s = sets[i]->getDescriptorSet(frameIndex, optionIndex);
        if (s == VK_NULL_HANDLE)
            throw std::runtime_error("ComputePipeline::bindWithSets: VkDescriptorSet is null");

        vkSets.push_back(s);
    }

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
        static_cast<uint32_t>(vkSets.size()), vkSets.data(), 0, nullptr);
}