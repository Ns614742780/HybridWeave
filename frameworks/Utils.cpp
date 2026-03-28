#include "Utils.h"

#include <fstream>
#include <utility>

#include "Buffer.h"

namespace Utils {

    std::vector<const char*> stringVectorToCharPtrVector(const std::vector<std::string>& stringVector) {
        std::vector<const char*> charPtrVector;
        charPtrVector.reserve(stringVector.size());
        for (auto& s : stringVector) {
            charPtrVector.push_back(s.c_str());
        }
        return charPtrVector;
    }

    std::vector<char> readFile(const std::string& path) {
        std::vector<char> result;
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return result;
        }
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        result.resize(static_cast<size_t>(fileSize));
        file.read(result.data(), fileSize);

        if (!file) {
            result.clear();
        }
        file.close();
        return result;
    }


    Utils::BarrierBuilder& BarrierBuilder::queueFamilyIndex(uint32_t queueFamilyIndex) {
        this->_srcQueueFamilyIndex = queueFamilyIndex;
        this->_dstQueueFamilyIndex = queueFamilyIndex;
        return *this;
    }

    Utils::BarrierBuilder& BarrierBuilder::srcQueueFamilyIndex(uint32_t srcQueueFamilyIndex) {
        _srcQueueFamilyIndex = srcQueueFamilyIndex;
        return *this;
    }

    Utils::BarrierBuilder& BarrierBuilder::dstQueueFamilyIndex(uint32_t dstQueueFamilyIndex) {
        _dstQueueFamilyIndex = dstQueueFamilyIndex;
        return *this;
    }

    Utils::BarrierBuilder&
        BarrierBuilder::addBufferBarrier(const std::shared_ptr<Buffer>& buffer,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            uint32_t srcQueueFamilyIndex,
            uint32_t dstQueueFamilyIndex)
    {
        if (!buffer) {
            throw std::runtime_error("BarrierBuilder::addBufferBarrier: buffer is null");
        }

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
        barrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
        barrier.buffer = buffer->buffer;
        barrier.offset = 0;
        barrier.size = buffer->size;

        bufferMemoryBarriers.push_back(barrier);
        return *this;
    }

    Utils::BarrierBuilder&
        BarrierBuilder::addBufferBarrier(const std::shared_ptr<Buffer>& buffer,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask)
    {
        return addBufferBarrier(buffer,
            srcAccessMask,
            dstAccessMask,
            _srcQueueFamilyIndex,
            _dstQueueFamilyIndex);
    }

    Utils::BarrierBuilder&
        BarrierBuilder::addImageBarrier(VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkAccessFlags srcAccessMask,
            VkAccessFlags dstAccessMask,
            VkImageAspectFlags aspectMask,
            uint32_t baseMipLevel,
            uint32_t levelCount,
            uint32_t baseArrayLayer,
            uint32_t layerCount,
            uint32_t srcQueueFamilyIndex,
            uint32_t dstQueueFamilyIndex)
    {
        if (image == VK_NULL_HANDLE) {
            throw std::runtime_error("BarrierBuilder::addImageBarrier: image is null");
        }

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
        barrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
        barrier.image = image;

        barrier.subresourceRange.aspectMask = aspectMask;
        barrier.subresourceRange.baseMipLevel = baseMipLevel;
        barrier.subresourceRange.levelCount = levelCount;
        barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
        barrier.subresourceRange.layerCount = layerCount;

        imageMemoryBarriers.push_back(barrier);
        return *this;
    }

    void BarrierBuilder::build(VkCommandBuffer commandBuffer,
        VkPipelineStageFlags srcStageMask,
        VkPipelineStageFlags dstStageMask) const
    {
        vkCmdPipelineBarrier(
            commandBuffer,
            srcStageMask,
            dstStageMask,
            0,
            0, nullptr,
            static_cast<uint32_t>(bufferMemoryBarriers.size()),
            bufferMemoryBarriers.data(),
            static_cast<uint32_t>(imageMemoryBarriers.size()),
            imageMemoryBarriers.data()
        );
    }

} // namespace Utils
