#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "VulkanContext.h"
#include "Buffer.h"

struct PlyProperty {
    std::string type;
    std::string name;
};

struct PlyHeader {
    std::string format;
    int numVertices = 0;
    int numFaces = 0;
    std::vector<PlyProperty> vertexProperties;
    std::vector<PlyProperty> faceProperties;
};

class GSScene {
public:
    explicit GSScene(const std::string& filename)
        : filename(filename) {
    }

    void load(const std::shared_ptr<VulkanContext>& context);
    void loadTestScene(const std::shared_ptr<VulkanContext>& context);

    // optional helper
    uint64_t getNumVertices() const { return static_cast<uint64_t>(header.numVertices); }

    struct Vertex {
        glm::vec4 position;
        glm::vec4 scale_opacity;
        glm::vec4 rotation;
        float     shs[48];
    };

    std::shared_ptr<Buffer> vertexBuffer;
    std::shared_ptr<Buffer> cov3DBuffer;

private:
    std::string filename;
    PlyHeader header;

    void loadPlyHeader(std::ifstream& plyFile);

    std::shared_ptr<Buffer> createBuffer(const std::shared_ptr<VulkanContext>& context, size_t size);

    void precomputeCov3D(const std::shared_ptr<VulkanContext>& context);
};
