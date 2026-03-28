#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <array>
#include <memory>
#include <stdexcept>

#include <vulkan/vulkan.h>

class Buffer;

namespace Utils {

    std::vector<const char*> stringVectorToCharPtrVector(const std::vector<std::string>& stringVector);
    std::vector<char> readFile(const std::string& path);

    template<uint32_t N>
    inline std::vector<glm::vec<N, float, glm::defaultp>>
        zipVectors(const std::array<std::vector<float>, N>& vectors) {
        std::vector<glm::vec<N, float, glm::defaultp>> result;
        for (uint32_t i = 0; i < vectors[0].size(); i++) {
            glm::vec<N, float, glm::defaultp> vec;
            for (uint32_t j = 0; j < N; j++) {
                vec[j] = vectors[j][i];
            }
            result.push_back(vec);
        }
        return result;
    }

    template<uint32_t N>
    inline std::vector<float>
        zipAndFlattenVectors(std::array<std::vector<float>, 48> arrays) {
        std::vector<float> result;
        for (uint32_t i = 0; i < arrays[0].size(); i++) {
            for (uint32_t j = 0; j < N; j++) {
                result.push_back(arrays[j][i]);
            }
        }
        return result;
    }

    class BarrierBuilder {
    public:
        BarrierBuilder& queueFamilyIndex(uint32_t queueFamilyIndex);

        BarrierBuilder& srcQueueFamilyIndex(uint32_t srcQueueFamilyIndex);
        BarrierBuilder& dstQueueFamilyIndex(uint32_t dstQueueFamilyIndex);

        // Buffer barriers
        BarrierBuilder& addBufferBarrier(const std::shared_ptr<Buffer>& buffer,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            uint32_t srcQueueFamilyIndex,
            uint32_t dstQueueFamilyIndex);

        BarrierBuilder& addBufferBarrier(const std::shared_ptr<Buffer>& buffer,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask);

        BarrierBuilder& addImageBarrier(VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            uint32_t baseMipLevel = 0,
            uint32_t levelCount = 1,
            uint32_t baseArrayLayer = 0,
            uint32_t layerCount = 1,
            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

        void build(VkCommandBuffer commandBuffer,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask) const;

    private:
        std::vector<VkBufferMemoryBarrier> bufferMemoryBarriers;
        std::vector<VkImageMemoryBarrier>  imageMemoryBarriers;

        uint32_t _srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        uint32_t _dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    };
}
