#pragma once
#include <vector>
#include <cstdint>

struct Image;

class IGBufferProvider {
public:
    virtual ~IGBufferProvider() = default;

    struct GBufferImages {
        std::shared_ptr<Image> albedoRough;
        std::shared_ptr<Image> normalMetal;
        std::shared_ptr<Image> worldPos;
        std::shared_ptr<Image> emissiveAO;
        std::shared_ptr<Image> sheenColorRough;
        std::shared_ptr<Image> material;
        std::shared_ptr<Image> drawId;
        std::shared_ptr<Image> depth;
    };

    virtual const std::vector<GBufferImages>& getGBufferImages() const = 0;
    virtual const uint32_t getGBufferMipLevels() const = 0;
};
