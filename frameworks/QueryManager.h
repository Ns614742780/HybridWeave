#pragma once

#include <vulkan/vulkan.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>
#include <mutex>

class QueryManager {
public:
    QueryManager(
        VkDevice device,
        VkQueryPool queryPool,
        uint32_t maxQueriesPerFrame,
        uint32_t framesInFlight);

    // 在 vkBeginCommandBuffer 之后调用
    void beginFrame(uint32_t frameIndex, VkCommandBuffer cmd);

    // Pass 在 record() 中调用（写 timestamp）
    void writeTimestamp(
        uint32_t frameIndex,
        VkCommandBuffer cmd,
        VkPipelineStageFlagBits stage,
        const std::string& name);

    // fence 已 signal 后调用（读取并打印）
    void resolveAndPrint(uint32_t frameIndex);

private:
    struct FrameState {
        uint32_t nextId = 0;
        uint32_t usedCount = 0;
        std::unordered_map<std::string, uint32_t> registry;
    };

    uint32_t registerQuery(uint32_t frameIndex, const std::string& name);

    std::unordered_map<std::string, uint64_t>
        parseResults(uint32_t frameIndex,
            const std::vector<uint64_t>& timestamps);

private:
    VkDevice device = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;

    uint32_t maxQueriesPerFrame = 0;
    uint32_t framesInFlight = 0;

    std::vector<FrameState> frames;

    std::mutex mutex;
};
