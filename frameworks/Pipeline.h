#pragma once

#include <memory>
#include <map>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#include "VulkanContext.h"
#include "DescriptorSet.h"

class Pipeline {
public:
    struct DescriptorOption {
        bool multiple;
        uint32_t value;
        std::vector<uint32_t> values;

        DescriptorOption(uint32_t value) : multiple(false), value(value) {}
        DescriptorOption(std::vector<uint32_t> values)
            : multiple(true), value(0), values(std::move(values)) {
        }

        [[nodiscard]] uint32_t get(size_t index) const;
    };

    explicit Pipeline(const std::shared_ptr<VulkanContext>& context);

    Pipeline(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;

    virtual ~Pipeline();

    void addDescriptorSet(uint32_t set, std::shared_ptr<DescriptorSet> descriptorSet);

    void addPushConstant(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size);

    virtual void build() = 0;

    virtual void bind(VkCommandBuffer commandBuffer,
        uint8_t currentFrame,
        DescriptorOption option);

    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }

protected:
    void buildPipelineLayout();

    void ensureContiguousDescriptorSets();

    std::shared_ptr<VulkanContext> context;

    std::vector<VkPushConstantRange> pushConstantRanges;

    std::map<uint32_t, std::shared_ptr<DescriptorSet>> descriptorSets;

    std::vector<std::shared_ptr<DescriptorSet>> contiguousSets; // size = maxSet+1, index == set
    uint32_t contiguousMaxSet = 0;

    std::vector<std::shared_ptr<DescriptorSet>> ownedFillerSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};
