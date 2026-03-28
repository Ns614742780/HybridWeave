#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <stdexcept>

#include <vulkan/vulkan.h>
#include "VulkanContext.h"
#include "vk_mem_alloc.h"

class DescriptorSet;

class Buffer : public std::enable_shared_from_this<Buffer> {
public:
    Buffer(const std::shared_ptr<VulkanContext>& context,
        uint32_t size,
        VkBufferUsageFlags usage,
        VmaMemoryUsage vmaUsage,
        VmaAllocationCreateFlags flags,
        bool concurrentSharing = false,
        VkDeviceSize alignment = 0,
        std::string debugName = "Unnamed");

    Buffer(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    ~Buffer();

    void realloc(uint64_t newSize);

    void boundToDescriptorSet(std::weak_ptr<DescriptorSet> descriptorSet,
        uint32_t set,
        uint32_t binding,
        VkDescriptorType type);

    static std::shared_ptr<Buffer> uniform(std::shared_ptr<VulkanContext> context,
        uint32_t size,
        bool concurrentSharing = false);

    static std::shared_ptr<Buffer> staging(std::shared_ptr<VulkanContext> context,
        unsigned long size);

    static std::shared_ptr<Buffer> storage(std::shared_ptr<VulkanContext> context,
        uint64_t size,
        bool concurrentSharing = false,
        VkDeviceSize alignment = 0,
        std::string debugName = "Unnamed Storage Buffer");

    static std::shared_ptr<Buffer> vertex(
        std::shared_ptr<VulkanContext> context,
        uint64_t size,
        bool concurrentSharing = false,
        std::string debugName = {}
    );

    static std::shared_ptr<Buffer> index(
        std::shared_ptr<VulkanContext> context,
        uint64_t size,
        bool concurrentSharing = false,
        std::string debugName = {}
    );

    void upload(const void* data, uint32_t size, uint32_t offset = 0);
    void uploadFrom(std::shared_ptr<Buffer> buffer);

    std::vector<char> download();
    void downloadTo(std::shared_ptr<Buffer> buffer,
        VkDeviceSize srcOffset = 0,
        VkDeviceSize dstOffset = 0);

    void assertEquals(char* data, size_t length);

    template<typename T>
    T readOne(VkDeviceSize offset = 0) {
        if (vmaUsage == VMA_MEMORY_USAGE_GPU_ONLY ||
            vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {
            const auto stagingBuffer = Buffer::staging(context, sizeof(T));
            downloadTo(stagingBuffer, offset, 0);
            return *static_cast<T*>(stagingBuffer->allocation_info.pMappedData);
        }
        else if (flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
            vmaInvalidateAllocation(context->allocator, allocation, offset, sizeof(T));
            return *(static_cast<T*>(allocation_info.pMappedData) + offset / sizeof(T));
        }
        else {
            throw std::runtime_error("Buffer is not mappable");
        }
    }

    void computeWriteReadBarrier(VkCommandBuffer commandBuffer);
    void computeReadWriteBarrier(VkCommandBuffer commandBuffer);
    void computeWriteWriteBarrier(VkCommandBuffer commandBuffer);

    VkDeviceSize size;
    VkBufferUsageFlags usage;
    uint64_t alignment;
    bool shared;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocation_info{};

    VmaMemoryUsage vmaUsage;
    VmaAllocationCreateFlags flags;

private:
    void alloc();
    Buffer createStagingBuffer(uint32_t size);

    std::shared_ptr<VulkanContext> context;

    std::vector<std::tuple<std::weak_ptr<DescriptorSet>, uint32_t, uint32_t, VkDescriptorType>> boundDescriptorSets;
    std::string debugName;
};
