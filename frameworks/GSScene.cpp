#include "GSScene.h"

#include <chrono>
#include <random>
#include <sstream>
#include <stdexcept>

#include "Utils.h"
#include "DescriptorSet.h"
#include "ComputePipeline.h"
#include "Shader.h"

#include "spdlog/spdlog.h"

struct VertexStorage {
    glm::vec3 position;
    glm::vec3 normal;
    float shs[48];
    float opacity;
    glm::vec3 scale;
    glm::vec4 rotation;
};


static_assert(sizeof(VertexStorage) == 62 * sizeof(float),
    "VertexStorage must be tightly packed (62 floats). Check struct packing / PLY layout.");


void GSScene::load(const std::shared_ptr<VulkanContext>& context)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream plyFile(filename, std::ios::binary);
    loadPlyHeader(plyFile);

    if (header.numVertices <= 0) {
        throw std::runtime_error("GSScene::load: invalid numVertices in PLY header");
    }

    vertexBuffer = createBuffer(context, static_cast<size_t>(header.numVertices) * sizeof(Vertex));

    auto staging = Buffer::staging(context, static_cast<unsigned long>(header.numVertices) * sizeof(Vertex));
    auto* verts = reinterpret_cast<Vertex*>(staging->allocation_info.pMappedData);

    for (int i = 0; i < header.numVertices; i++) {
        VertexStorage storage{};
        plyFile.read(reinterpret_cast<char*>(&storage), sizeof(VertexStorage));
        if (!plyFile) {
            throw std::runtime_error("GSScene::load: failed reading VertexStorage from PLY (EOF or IO error)");
        }

        verts[i].position = glm::vec4(storage.position, 1.0f);

        verts[i].scale_opacity =
            glm::vec4(glm::exp(storage.scale),
                1.0f / (1.0f + std::exp(-storage.opacity)));

        verts[i].rotation = glm::normalize(storage.rotation);

        verts[i].shs[0] = storage.shs[0];
        verts[i].shs[1] = storage.shs[1];
        verts[i].shs[2] = storage.shs[2];

        for (int j = 0; j < 15; j++) {
            verts[i].shs[3 + j * 3 + 0] = storage.shs[3 + j + 0 * 15];
            verts[i].shs[3 + j * 3 + 1] = storage.shs[3 + j + 1 * 15];
            verts[i].shs[3 + j * 3 + 2] = storage.shs[3 + j + 2 * 15];
        }
    }

    vertexBuffer->uploadFrom(staging);

    auto end = std::chrono::high_resolution_clock::now();
    spdlog::info("Loaded GSScene: {} verts, time={}ms",
        header.numVertices,
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    precomputeCov3D(context);
}

void GSScene::loadTestScene(const std::shared_ptr<VulkanContext>& context)
{
    header.numVertices = 1;

    vertexBuffer = createBuffer(context, sizeof(Vertex));

    auto staging = Buffer::staging(context, sizeof(Vertex));
    auto* verts = reinterpret_cast<Vertex*>(staging->allocation_info.pMappedData);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rnd(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scaleRnd(-2.0f, -1.0f);

    verts[0].position = glm::vec4(0, 0, 0, 1);
    verts[0].scale_opacity = glm::vec4(glm::exp(glm::vec3(scaleRnd(gen))), 1.0f);
    verts[0].rotation = glm::vec4(0, 0, 0, 1);

    for (int i = 0; i < 48; i++) verts[0].shs[i] = rnd(gen);

    vertexBuffer->uploadFrom(staging);

    precomputeCov3D(context);
}

void GSScene::loadPlyHeader(std::ifstream& ply)
{
    if (!ply.is_open()) {
        throw std::runtime_error("GSScene::loadPlyHeader: failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(ply, line)) {
        if (line == "end_header")
            break;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            iss >> header.format;
        }
        else if (token == "element") {
            iss >> token;
            if (token == "vertex") iss >> header.numVertices;
            else if (token == "face") iss >> header.numFaces;
        }
        else if (token == "property") {
            PlyProperty prop{};
            iss >> prop.type >> prop.name;

            if (header.numFaces == 0)
                header.vertexProperties.push_back(prop);
            else
                header.faceProperties.push_back(prop);
        }
    }

    if (header.format != "binary_little_endian") {
        throw std::runtime_error("GSScene::loadPlyHeader: only binary_little_endian PLY supported");
    }
}

std::shared_ptr<Buffer> GSScene::createBuffer(const std::shared_ptr<VulkanContext>& context, size_t size)
{
    return std::make_shared<Buffer>(
        context,
        static_cast<uint32_t>(size),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        false
    );
}

void GSScene::precomputeCov3D(const std::shared_ptr<VulkanContext>& context)
{
    cov3DBuffer = createBuffer(context, static_cast<size_t>(header.numVertices) * sizeof(float) * 6);

    auto shader = std::make_shared<Shader>(context, "precomp_cov3d");
    auto pipeline = std::make_shared<ComputePipeline>(context, shader);

    auto dset = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    dset->bindBufferToDescriptorSet(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, vertexBuffer);
    dset->bindBufferToDescriptorSet(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, cov3DBuffer);
    dset->build();

    pipeline->addDescriptorSet(0, dset);
    pipeline->addPushConstant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float));
    pipeline->build();

    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    Utils::BarrierBuilder()
        .addBufferBarrier(
            vertexBuffer,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT
        )
        .build(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        );

    pipeline->bind(cmd, 0, Pipeline::DescriptorOption(0));

    float factor = 1.0f;
    vkCmdPushConstants(
        cmd,
        pipeline->getPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(float),
        &factor
    );

    uint32_t groups = (static_cast<uint32_t>(header.numVertices) + 255u) / 256u;
    vkCmdDispatch(cmd, groups, 1, 1);

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::COMPUTE);

    spdlog::info("Precomputed Cov3D");
}
