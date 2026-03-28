#include "Buffer.h"

#include <iostream>
#include <utility>
#include <cstring>
#include <algorithm>

#include "Utils.h"
#include "DescriptorSet.h"
#include "spdlog/spdlog.h"

void Buffer::alloc() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = shared ? VK_SHARING_MODE_CONCURRENT
        : VK_SHARING_MODE_EXCLUSIVE;

    uint32_t queueFamilyIndices[2];
    uint32_t queueFamilyCount = 0;

    if (shared) {
        auto graphicsFamily = context->queues[VulkanContext::Queue::GRAPHICS].queueFamily;
        auto computeFamily = context->queues[VulkanContext::Queue::COMPUTE].queueFamily;
        queueFamilyIndices[0] = graphicsFamily;
        queueFamilyIndices[1] = computeFamily;
        queueFamilyCount = 2;

        bufferInfo.queueFamilyIndexCount = queueFamilyCount;
        bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
    }

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = vmaUsage;
    allocInfo.flags = flags;

    VkResult res;
    if (alignment != 0) {
        res = vmaCreateBufferWithAlignment(
            context->allocator,
            &bufferInfo,
            &allocInfo,
            alignment,
            &buffer,
            &allocation,
            &allocation_info
        );
    }
    else {
        res = vmaCreateBuffer(
            context->allocator,
            &bufferInfo,
            &allocInfo,
            &buffer,
            &allocation,
            &allocation_info
        );
    }

    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    // Debug name can be added here if needed (vkSetDebugUtilsObjectNameEXT)
}

Buffer::Buffer(const std::shared_ptr<VulkanContext>& _context,
    uint32_t size,
    VkBufferUsageFlags usage,
    VmaMemoryUsage vmaUsage,
    VmaAllocationCreateFlags flags,
    bool shared,
    VkDeviceSize alignment,
    std::string debugName)
    : context(_context),
    size(size),
    usage(usage),
    alignment(alignment),
    shared(shared),
    vmaUsage(vmaUsage),
    flags(flags),
    allocation(nullptr),
    debugName(std::move(debugName))
{
    alloc();
}

Buffer Buffer::createStagingBuffer(uint32_t size) {
    return Buffer(
        context,
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        false
    );
}

void Buffer::upload(const void* data, uint32_t size, uint32_t offset) {
    if (!data) {
        throw std::runtime_error("Buffer::upload: data is null");
    }
    if (size + offset > this->size) {
        throw std::runtime_error("Buffer::upload: Buffer overflow");
    }

    if (vmaUsage == VMA_MEMORY_USAGE_GPU_ONLY ||
        vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {

        auto stagingBuffer = createStagingBuffer(size);

        std::memcpy(
            stagingBuffer.allocation_info.pMappedData,
            data,
            size
        );

        VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = offset;
        copyRegion.size = size;

        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, buffer, 1, &copyRegion);
        context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::COMPUTE);
    }
    else if (flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
        std::memcpy(
            static_cast<char*>(allocation_info.pMappedData) + offset,
            data,
            size
        );
    }
    else {
        throw std::runtime_error("Buffer::upload: Buffer is not mappable");
    }
}

void Buffer::uploadFrom(std::shared_ptr<Buffer> src) {
    if (!src) {
        throw std::runtime_error("Buffer::uploadFrom: src is null");
    }
    if (src->size > size) {
        throw std::runtime_error("Buffer::uploadFrom: Buffer overflow");
    }

    if (vmaUsage == VMA_MEMORY_USAGE_GPU_ONLY ||
        vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {

        VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = src->size;

        vkCmdCopyBuffer(cmd, src->buffer, buffer, 1, &copyRegion);
        context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::COMPUTE);
    }
    else if (flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
        std::memcpy(allocation_info.pMappedData,
            src->allocation_info.pMappedData,
            src->size);
    }
    else {
        throw std::runtime_error("Buffer::uploadFrom: Buffer is not mappable");
    }
}

void Buffer::downloadTo(std::shared_ptr<Buffer> dst,
    VkDeviceSize srcOffset,
    VkDeviceSize dstOffset)
{
    if (!dst) {
        throw std::runtime_error("Buffer::downloadTo: dst is null");
    }

    const VkDeviceSize copySize = dst->size;

    if (srcOffset + copySize > this->size) {
        throw std::runtime_error("Buffer::downloadTo: src range out of bounds");
    }
    if (dstOffset + copySize > dst->size) {
        throw std::runtime_error("Buffer::downloadTo: dst range out of bounds");
    }

    if (vmaUsage == VMA_MEMORY_USAGE_GPU_ONLY ||
        vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {

        VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = srcOffset;
        copyRegion.dstOffset = dstOffset;
        copyRegion.size = copySize;

        vkCmdCopyBuffer(cmd, buffer, dst->buffer, 1, &copyRegion);
        context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::COMPUTE);
    }
    else if (flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
        std::memcpy(
            static_cast<char*>(dst->allocation_info.pMappedData) + dstOffset,
            static_cast<char*>(allocation_info.pMappedData) + srcOffset,
            static_cast<size_t>(copySize)
        );
    }
    else {
        throw std::runtime_error("Buffer::downloadTo: Buffer is not mappable");
    }
}

Buffer::~Buffer() {
    if (buffer != VK_NULL_HANDLE && allocation != nullptr) {
        vmaDestroyBuffer(context->allocator, buffer, allocation);
        spdlog::debug("Buffer destroyed: {}", debugName);
    }
}

void Buffer::realloc(uint64_t newSize) {
    if (buffer != VK_NULL_HANDLE && allocation != nullptr) {
        vmaDestroyBuffer(context->allocator, buffer, allocation);
    }

    size = newSize;
    alloc();

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = allocation_info.offset;
    bufferInfo.range = size;

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(boundDescriptorSets.size());

    for (auto& tup : boundDescriptorSets) {
        auto dsWeak = std::get<0>(tup);
        auto setIndex = std::get<1>(tup);
        auto binding = std::get<2>(tup);
        auto type = std::get<3>(tup);

        auto dsShared = dsWeak.lock();
        if (!dsShared) continue;

        if (setIndex >= dsShared->descriptorSets.size()) {
            continue;
        }

        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = dsShared->descriptorSets[setIndex];
        w.dstBinding = binding;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = type;
        w.pBufferInfo = &bufferInfo;
        w.pImageInfo = nullptr;

        writes.push_back(w);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(
            context->device,
            static_cast<uint32_t>(writes.size()), writes.data(),
            0, nullptr
        );
    }
}

void Buffer::boundToDescriptorSet(std::weak_ptr<DescriptorSet> descriptorSet,
    uint32_t set,
    uint32_t binding,
    VkDescriptorType type)
{
    boundDescriptorSets.emplace_back(descriptorSet, set, binding, type);
}

std::shared_ptr<Buffer> Buffer::uniform(std::shared_ptr<VulkanContext> context,
    uint32_t size,
    bool concurrentSharing)
{
    return std::make_shared<Buffer>(
        context,
        size,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        concurrentSharing
    );
}

std::shared_ptr<Buffer> Buffer::staging(std::shared_ptr<VulkanContext> context,
    unsigned long size)
{
    return std::make_shared<Buffer>(
        context,
        static_cast<uint32_t>(size),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        false
    );
}

std::shared_ptr<Buffer> Buffer::storage(std::shared_ptr<VulkanContext> context,
    uint64_t size,
    bool concurrentSharing,
    VkDeviceSize alignment,
    std::string debugName)
{
    return std::make_shared<Buffer>(
        context,
        static_cast<uint32_t>(size),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        concurrentSharing,
        alignment,
        std::move(debugName)
    );
}

std::shared_ptr<Buffer> Buffer::vertex(
    std::shared_ptr<VulkanContext> context,
    uint64_t size,
    bool concurrentSharing,
    std::string debugName)
{
    return std::make_shared<Buffer>(
        context,
        static_cast<uint32_t>(size),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        concurrentSharing,
        /*alignment*/ 0,
        std::move(debugName)
    );
}

std::shared_ptr<Buffer> Buffer::index(
    std::shared_ptr<VulkanContext> context,
    uint64_t size,
    bool concurrentSharing,
    std::string debugName)
{
    return std::make_shared<Buffer>(
        context,
        static_cast<uint32_t>(size),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        concurrentSharing,
        /*alignment*/ 0,
        std::move(debugName)
    );
}

void Buffer::assertEquals(char* data, size_t length) {
    if (!data) {
        throw std::runtime_error("Buffer::assertEquals: data is null");
    }
    if (length > size) {
        throw std::runtime_error("Buffer::assertEquals: Buffer overflow");
    }

    if (vmaUsage == VMA_MEMORY_USAGE_GPU_ONLY ||
        vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {

        auto stagingBuffer = Buffer::staging(context, static_cast<unsigned long>(length));
        downloadTo(stagingBuffer, 0, 0);

        if (std::memcmp(data, stagingBuffer->allocation_info.pMappedData, length) != 0) {
            throw std::runtime_error("Buffer content does not match");
        }
    }
    else if (flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
        if (std::memcmp(data, allocation_info.pMappedData, length) != 0) {
            throw std::runtime_error("Buffer content does not match");
        }
    }
    else {
        throw std::runtime_error("Buffer::assertEquals: Buffer is not mappable");
    }
}

void Buffer::computeWriteReadBarrier(VkCommandBuffer commandBuffer) {
    Utils::BarrierBuilder()
        .queueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .addBufferBarrier(shared_from_this(),
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT)
        .build(commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void Buffer::computeReadWriteBarrier(VkCommandBuffer commandBuffer) {
    Utils::BarrierBuilder()
        .queueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .addBufferBarrier(shared_from_this(),
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT)
        .build(commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void Buffer::computeWriteWriteBarrier(VkCommandBuffer commandBuffer) {
    Utils::BarrierBuilder()
        .queueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .addBufferBarrier(shared_from_this(),
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_WRITE_BIT)
        .build(commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

std::vector<char> Buffer::download() {
    auto stagingBuffer = Buffer::staging(context, static_cast<unsigned long>(size));
    downloadTo(stagingBuffer, 0, 0);
    char* begin = static_cast<char*>(stagingBuffer->allocation_info.pMappedData);
    return std::vector<char>(begin, begin + size);
}
