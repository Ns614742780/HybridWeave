#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

#include "VulkanContext.h"
#include "Buffer.h"
#include "Swapchain.h"

class Buffer;
struct Image;

class DescriptorSet : public std::enable_shared_from_this<DescriptorSet> {
public:
    struct DescriptorBinding {
        VkDescriptorType              type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        VkDescriptorSetLayoutBinding  layoutBinding{};

        std::shared_ptr<Buffer>       buffer;
        VkDescriptorBufferInfo        bufferInfo{};

        std::shared_ptr<Image>        image;
        VkDescriptorImageInfo         imageInfo{};

        std::vector<std::shared_ptr<Image>> images;
        std::vector<VkDescriptorImageInfo>  imageInfos;
        VkDescriptorImageInfo rawImageInfo{};

        VkImageView rawImageView = VK_NULL_HANDLE;
        VkSampler   rawSampler = VK_NULL_HANDLE;
        VkImageLayout rawLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    };

    DescriptorSet(const std::shared_ptr<VulkanContext>& context, uint8_t framesInFlight = 1);
    ~DescriptorSet();

    void bindBufferToDescriptorSet(uint32_t binding,
        VkDescriptorType type,
        VkShaderStageFlags stage,
        std::shared_ptr<Buffer> buffer);

    void bindImageToDescriptorSet(uint32_t binding,
        VkDescriptorType type,
        VkShaderStageFlags stage,
        std::shared_ptr<Image> image);

    void bindCombinedImageSamplerToDescriptorSet(
        uint32_t binding,
        VkShaderStageFlags stageFlags,
        std::shared_ptr<Image> image,
        VkSampler sampler,
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    void bindCombinedImageSamplerArrayToDescriptorSet(
        uint32_t binding,
        VkShaderStageFlags stageFlags,
        const std::vector<std::shared_ptr<Image>>& images,
        const std::vector<VkSampler>& samplers,
        VkImageLayout imageLayout);

    void updateCombinedImageSampler(
        uint32_t binding,
        uint32_t frameIndex,
        const std::shared_ptr<Image>& image,
        VkSampler sampler,
        VkImageLayout layout);

    void bindCombinedImageSamplerRaw(
        uint32_t binding,
        VkShaderStageFlags stageFlags,
        VkImageView imageView,
        VkSampler sampler,
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    void bindStorageImageToDescriptorSet(
        uint32_t binding,
        VkShaderStageFlags stageFlags,
        std::shared_ptr<Image> image,
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_GENERAL);

    void updateStorageImage(
        uint32_t binding,
        uint32_t frameIndex,
        const std::shared_ptr<Image>& image,
        VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);

    void build();

    VkDescriptorSet getDescriptorSet(uint8_t frame, uint32_t option) const;

public:
    VkDescriptorSetLayout              descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet>       descriptorSets;
    size_t                             maxOptions = 1;

private:
    std::shared_ptr<VulkanContext>     context;
    uint8_t                            framesInFlight = 1;

    std::unordered_map<uint32_t, std::vector<DescriptorBinding>> bindings;
};
