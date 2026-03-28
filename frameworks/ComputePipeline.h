#pragma once

#include "Pipeline.h"
#include "Shader.h"

class ComputePipeline : public Pipeline {
public:
    explicit ComputePipeline(const std::shared_ptr<VulkanContext>& context,
        std::shared_ptr<Shader> shader);

    void build() override;
    void bindWithSet(VkCommandBuffer cmd, uint8_t frameIndex, uint32_t optionIndex, std::shared_ptr<DescriptorSet> set);
    void bindWithSets(VkCommandBuffer cmd, uint8_t frameIndex, uint32_t optionIndex, const std::vector<std::shared_ptr<DescriptorSet>>& sets);
private:
    std::shared_ptr<Shader> shader;
};
