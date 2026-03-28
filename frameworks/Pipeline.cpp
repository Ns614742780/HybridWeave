#include "Pipeline.h"

#include <utility>

uint32_t Pipeline::DescriptorOption::get(size_t index) const {
    return multiple ? values[index] : value;
}

Pipeline::Pipeline(const std::shared_ptr<VulkanContext>& _context)
    : context(_context)
{
    if (!context) {
        throw std::runtime_error("Pipeline: context is null");
    }
}

Pipeline::~Pipeline()
{
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context->device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }

    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context->device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }

    contiguousSets.clear();
    ownedFillerSets.clear();
}

void Pipeline::addDescriptorSet(uint32_t set, std::shared_ptr<DescriptorSet> descriptorSet) {
    if (!descriptorSet) {
        throw std::runtime_error("Pipeline::addDescriptorSet: descriptorSet is null");
    }
    descriptorSets[set] = std::move(descriptorSet);
}

void Pipeline::addPushConstant(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size)
{
    VkPushConstantRange range{};
    range.stageFlags = stageFlags;
    range.offset = offset;
    range.size = size;
    pushConstantRanges.push_back(range);
}

void Pipeline::ensureContiguousDescriptorSets()
{
    contiguousSets.clear();
    ownedFillerSets.clear();

    if (descriptorSets.empty()) {
        contiguousMaxSet = 0;
        auto filler = std::make_shared<DescriptorSet>(context, /*framesInFlight*/1);
        filler->build();
        ownedFillerSets.push_back(filler);
        contiguousSets.push_back(filler);
        return;
    }

    contiguousMaxSet = descriptorSets.rbegin()->first;
    contiguousSets.resize(static_cast<size_t>(contiguousMaxSet) + 1);

    for (auto& kv : descriptorSets) {
        contiguousSets[kv.first] = kv.second;
    }

    uint8_t framesInFlight = 1;
    for (uint32_t set = 0; set <= contiguousMaxSet; ++set) {
        if (!contiguousSets[set]) {
            auto filler = std::make_shared<DescriptorSet>(context, framesInFlight);
            filler->build();
            ownedFillerSets.push_back(filler);
            contiguousSets[set] = filler;
        }
    }
}

void Pipeline::buildPipelineLayout()
{
    ensureContiguousDescriptorSets();

    std::vector<VkDescriptorSetLayout> layouts;
    layouts.reserve(contiguousSets.size());

    for (uint32_t set = 0; set <= contiguousMaxSet; ++set) {
        auto& ds = contiguousSets[set];
        if (!ds || ds->descriptorSetLayout == VK_NULL_HANDLE) {
            throw std::runtime_error("Pipeline::buildPipelineLayout: descriptor set layout is null");
        }
        layouts.push_back(ds->descriptorSetLayout);
    }

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
    layoutInfo.pSetLayouts = layouts.data();

    if (!pushConstantRanges.empty()) {
        layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
        layoutInfo.pPushConstantRanges = pushConstantRanges.data();
    }

    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context->device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }

    if (vkCreatePipelineLayout(context->device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout.");
    }
}

void Pipeline::bind(VkCommandBuffer commandBuffer,
    uint8_t currentFrame,
    DescriptorOption option)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    if (contiguousSets.empty()) {
        ensureContiguousDescriptorSets();
    }

    std::vector<VkDescriptorSet> sets;
    sets.reserve(static_cast<size_t>(contiguousMaxSet) + 1);

    size_t optIdx = 0;

    for (uint32_t set = 0; set <= contiguousMaxSet; ++set) {
        auto& ds = contiguousSets[set];
        if (!ds) {
            throw std::runtime_error("Pipeline::bind: contiguous descriptor set is null");
        }

        bool isReal = (descriptorSets.find(set) != descriptorSets.end());
        uint32_t opt = 0;

        if (isReal) {
            if (option.multiple) {
                if (optIdx >= option.values.size()) {
                    throw std::runtime_error("Pipeline::bind: DescriptorOption.values is too small for real descriptor sets");
                }
            }
            opt = option.get(optIdx++);
        }

        sets.push_back(ds->getDescriptorSet(currentFrame, opt));
    }

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0, // firstSet
        static_cast<uint32_t>(sets.size()),
        sets.data(),
        0,
        nullptr
    );
}