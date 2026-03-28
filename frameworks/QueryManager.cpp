#include "QueryManager.h"

#include <stdexcept>
#include <iostream>

QueryManager::QueryManager(
    VkDevice device_,
    VkQueryPool queryPool_,
    uint32_t maxQueriesPerFrame_,
    uint32_t framesInFlight_)
    : device(device_)
    , queryPool(queryPool_)
    , maxQueriesPerFrame(maxQueriesPerFrame_)
    , framesInFlight(framesInFlight_)
{
    frames.resize(framesInFlight);
}

void QueryManager::beginFrame(uint32_t frameIndex, VkCommandBuffer cmd)
{
    std::lock_guard<std::mutex> lock(mutex);

    FrameState& f = frames[frameIndex];
    f.registry.clear();
    f.nextId = 0;
    f.usedCount = 0;

    vkCmdResetQueryPool(
        cmd,
        queryPool,
        0,
        maxQueriesPerFrame);
}

uint32_t QueryManager::registerQuery(uint32_t frameIndex,
    const std::string& name)
{
    FrameState& f = frames[frameIndex];

    auto it = f.registry.find(name);
    if (it != f.registry.end()) {
        return it->second;
    }

    if (f.nextId >= maxQueriesPerFrame) {
        throw std::runtime_error(
            "QueryManager: exceeded maxQueriesPerFrame");
    }

    uint32_t id = f.nextId++;
    f.registry[name] = id;
    f.usedCount = f.nextId;
    return id;
}

void QueryManager::writeTimestamp(
    uint32_t frameIndex,
    VkCommandBuffer cmd,
    VkPipelineStageFlagBits stage,
    const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex);

    uint32_t id = registerQuery(frameIndex, name);

    vkCmdWriteTimestamp(
        cmd,
        stage,
        queryPool,
        id);
}

void QueryManager::resolveAndPrint(uint32_t frameIndex)
{
    FrameState& f = frames[frameIndex];
    if (f.usedCount == 0) {
        return;
    }

    std::vector<uint64_t> timestamps(f.usedCount);

    VkResult res = vkGetQueryPoolResults(
        device,
        queryPool,
        0,
        f.usedCount,
        timestamps.size() * sizeof(uint64_t),
        timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);

    if (res == VK_NOT_READY) {
        return;
    }

    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve timestamps");
    }

    auto metrics = parseResults(frameIndex, timestamps);

    for (const auto& m : metrics) {
        std::cout
            << "[GPU] " << m.first << ": "
            << (m.second / 1e6) << " ms\n";
    }
}

std::unordered_map<std::string, uint64_t>
QueryManager::parseResults(
    uint32_t frameIndex,
    const std::vector<uint64_t>& timestamps)
{
    FrameState& f = frames[frameIndex];
    std::unordered_map<std::string, uint64_t> results;

    for (const auto& kv : f.registry) {
        const std::string& name = kv.first;
        uint32_t startId = kv.second;

        const std::string suffix = "_start";
        if (name.size() < suffix.size()) {
            continue;
        }

        if (name.compare(
            name.size() - suffix.size(),
            suffix.size(),
            suffix) != 0) {
            continue;
        }

        std::string base = name.substr(
            0, name.size() - suffix.size());
        std::string endName = base + "_end";

        auto itEnd = f.registry.find(endName);
        if (itEnd == f.registry.end()) {
            continue;
        }

        uint32_t endId = itEnd->second;

        if (startId >= timestamps.size() ||
            endId >= timestamps.size()) {
            continue;
        }

        results[base] = timestamps[endId] - timestamps[startId];
    }

    return results;
}
