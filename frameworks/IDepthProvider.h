#pragma once
#include <vector>
#include <cstdint>

struct Image;

class IDepthProvider {
public:
    virtual ~IDepthProvider() = default;
    virtual std::shared_ptr<Image> getDepthImage(uint32_t imageIndex) const = 0;
};
