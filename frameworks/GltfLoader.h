#pragma once
#include <tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <unordered_map>

class VulkanContext;
class Buffer;

enum MaterialFlagBits : uint8_t
{
	MATERIAL_FLAG_CLOTH = 1 << 0,  // cloth
	MATERIAL_FLAG_FOLIAGE = 1 << 1,  // plant
    MATERIAL_FLAG_SHEEN = 1 << 2,  // sheen BRDF
	MATERIAL_FLAG_THIN = 1 << 3,  // leaf / cloth litting
	MATERIAL_FLAG_TRANSMISSION = 1 << 4,  // transmission material like glass
	MATERIAL_FLAG_DOUBLE_SIDED = 1 << 5,  // double sided
	MATERIAL_FLAG_ALPHA_MASK = 1 << 6,  // alpha mask
};

struct GltfDrawItem {
    uint32_t meshGpuIndex; //  meshes[meshGpuIndex]
    glm::mat4 world;       // node's world matrix
    uint32_t drawId;
};

struct GltfTextureGPU {
    VkImage        image = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    VkSampler      sampler = VK_NULL_HANDLE;
    VkFormat       format = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t       width = 0;
    uint32_t       height = 0;
    void* allocation = nullptr; // VmaAllocation
};

struct GltfMeshGPU {
    std::shared_ptr<Buffer> vertexBuffer;
    std::shared_ptr<Buffer> indexBuffer;
    uint32_t  indexCount = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    glm::mat4 model = glm::mat4(1.0f);

    // ---- Base Color ----
    bool hasBaseColorTex = false;
    int  baseColorTex = -1;               // image index
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    uint8_t uvSetBaseColor = 0;

    // ---- Metallic / Roughness ----
    bool hasMetallicRoughnessTex = false;
    int  metallicRoughnessTex = -1;       // image index
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    uint8_t uvSetMR = 0;

    // ---- Normal map ----
    bool hasNormalTex = false;
    int  normalTex = -1;                  // image index
    float normalScale = 1.0f;
    uint8_t uvSetNormal = 0;

    // ---- Occlusion (AO) ----
    bool hasOcclusionTex = false;
    int  occlusionTex = -1;               // image index
    float occlusionStrength = 1.0f;
    uint8_t uvSetOcclusion = 0;

    // ---- Emissive ----
    bool hasEmissiveTex = false;
    int  emissiveTex = -1;                // image index
    glm::vec3 emissiveFactor = glm::vec3(0.0f);
    uint8_t uvSetEmissive = 0;

    // ---- Sheen (KHR_materials_sheen) ----
    bool hasSheen = false;
    int  sheenColorTex = -1;
    glm::vec3 sheenColorFactor = glm::vec3(0.0f);
    float sheenRoughnessFactor = 0.0f;
    uint8_t materialFlags = 0;
    uint8_t uvSetSheenColor = 0;

    bool hasSheenRoughnessTex = false;
    int sheenRoughnessTex = -1;
    uint8_t uvSetSheenRoughness = 0;

    // ---- Transmission (KHR_materials_transmission) ----
    bool  hasTransmission = false;
    float transmissionFactor = 0.0f;
    int   transmissionTex = -1;
    uint8_t uvSetTransmission = 0;

    // ---- Tangents for normal map ----
    bool hasTangents = false;
};

class GltfLoaderVulkan {
public:
    struct RootTransform {
        glm::vec3 translate = glm::vec3(0.0f);
        glm::quat rotate = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // w,x,y,z
        glm::vec3 scale = glm::vec3(1.0f);
    };

    explicit GltfLoaderVulkan(std::shared_ptr<VulkanContext> ctx);

    bool loadModel(const std::string& path);
    void destroy(); // free GPU resources
    void dumpSummary() const; // compare this with your OpenGL version

    void setRootTransform(const RootTransform& t) { rootXform = t; }

    const RootTransform& getRootTransform() const { return rootXform; }
    const std::vector<GltfMeshGPU>& getMeshes() const { return meshes; }
    const std::vector<GltfTextureGPU>& getTextures() const { return textures; }
    const std::vector<GltfDrawItem>& getDrawItems() const { return drawItems; }

    struct TextureKey {
        int imageIndex;
        bool srgb;

        bool operator==(const TextureKey& o) const {
            return imageIndex == o.imageIndex && srgb == o.srgb;
        }
    };

    struct TextureKeyHash {
        size_t operator()(const TextureKey& k) const {
            return std::hash<int>()(k.imageIndex) ^ (k.srgb ? 0x9e3779b9 : 0);
        }
    };

private:
    // --- tinygltf helpers
    const uint8_t* accessorDataPtr(const tinygltf::Model& model, const tinygltf::Accessor& acc) const;

    void buildDrawItemsFromScene();

    // --- texture creation ---
    GltfTextureGPU createTextureFromImage(const tinygltf::Image& image, bool srgb);
    void destroyTexture(GltfTextureGPU& t);

    int getOrCreateTexture(int imageIndex, bool srgb);

private:
    std::shared_ptr<VulkanContext> context;
    std::vector<GltfTextureGPU> textures;   // index = image index
    std::vector<GltfMeshGPU> meshes;
    std::vector<GltfDrawItem> drawItems;

    // glTF mesh index -> gpu mesh indices (one per primitive)
    std::unordered_map<int, std::vector<uint32_t>> gltfMeshIndexToGpuMeshes;
    std::unordered_map<TextureKey, int, TextureKeyHash> textureCache;

    tinygltf::Model gltfModel;
    RootTransform rootXform;
};
