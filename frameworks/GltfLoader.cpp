#include "GltfLoader.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "VulkanContext.h"
#include "Buffer.h"

#include "vk_mem_alloc.h"
#include <spdlog/spdlog.h>

static size_t componentSize(int componentType) {
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE:           return 1;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  return 1;
    case TINYGLTF_COMPONENT_TYPE_SHORT:          return 2;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
    case TINYGLTF_COMPONENT_TYPE_INT:            return 4;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   return 4;
    case TINYGLTF_COMPONENT_TYPE_FLOAT:          return 4;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:         return 8;
    default: return 0;
    }
}

static int typeCount(int type) {
    switch (type) {
    case TINYGLTF_TYPE_SCALAR: return 1;
    case TINYGLTF_TYPE_VEC2:   return 2;
    case TINYGLTF_TYPE_VEC3:   return 3;
    case TINYGLTF_TYPE_VEC4:   return 4;
    case TINYGLTF_TYPE_MAT4:   return 16;
    default: return 0;
    }
}

static glm::mat4 quatToMat4(const glm::quat& q)
{
    return glm::mat4(glm::mat3_cast(q));
}

static size_t accessorStride(const tinygltf::Model& model, const tinygltf::Accessor& acc) {
    const auto& view = model.bufferViews[acc.bufferView];
    if (view.byteStride != 0) return (size_t)view.byteStride;
    return componentSize(acc.componentType) * (size_t)typeCount(acc.type);
}

static glm::mat4 nodeLocalMatrix(const tinygltf::Node& n)
{
    if (n.matrix.size() == 16) {
        glm::mat4 m(1.0f);
        const double* a = n.matrix.data();
        m[0][0] = (float)a[0];  m[0][1] = (float)a[1];  m[0][2] = (float)a[2];  m[0][3] = (float)a[3];
        m[1][0] = (float)a[4];  m[1][1] = (float)a[5];  m[1][2] = (float)a[6];  m[1][3] = (float)a[7];
        m[2][0] = (float)a[8];  m[2][1] = (float)a[9];  m[2][2] = (float)a[10]; m[2][3] = (float)a[11];
        m[3][0] = (float)a[12]; m[3][1] = (float)a[13]; m[3][2] = (float)a[14]; m[3][3] = (float)a[15];
        return m;
    }

    glm::vec3 t(0.0f), s(1.0f);
    glm::quat r(1.0f, 0.0f, 0.0f, 0.0f);

    if (n.translation.size() == 3)
        t = glm::vec3((float)n.translation[0], (float)n.translation[1], (float)n.translation[2]);

    if (n.scale.size() == 3)
        s = glm::vec3((float)n.scale[0], (float)n.scale[1], (float)n.scale[2]);

    if (n.rotation.size() == 4)
        r = glm::quat((float)n.rotation[3], (float)n.rotation[0], (float)n.rotation[1], (float)n.rotation[2]); // (w,x,y,z)

    glm::mat4 M = glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.0f), s);
    return M;
}

static glm::mat4 makeRootMatrix(const GltfLoaderVulkan::RootTransform& t)
{
    glm::mat4 T = glm::translate(glm::mat4(1.0f), t.translate);
    glm::mat4 R = quatToMat4(glm::normalize(t.rotate));
    glm::mat4 S = glm::scale(glm::mat4(1.0f), t.scale);
    return T * R * S;
}

static void cmdTransitionImage(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}


GltfLoaderVulkan::GltfLoaderVulkan(std::shared_ptr<VulkanContext> ctx)
    : context(std::move(ctx))
{
    if (!context) throw std::runtime_error("GltfLoaderVulkan: context is null");
}

const uint8_t* GltfLoaderVulkan::accessorDataPtr(const tinygltf::Model& model, const tinygltf::Accessor& acc) const
{
    if (acc.bufferView < 0) throw std::runtime_error("accessorDataPtr: accessor has no bufferView");
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[view.buffer];

    size_t offset = (size_t)view.byteOffset + (size_t)acc.byteOffset;
    if (offset >= buf.data.size()) throw std::runtime_error("accessorDataPtr: offset out of range");

    return buf.data.data() + offset;
}

GltfTextureGPU GltfLoaderVulkan::createTextureFromImage(const tinygltf::Image& image, bool srgb)
{
    if (image.width <= 0 || image.height <= 0)
        throw std::runtime_error("createTextureFromImage: invalid image size");

    const int w = image.width;
    const int h = image.height;

    std::vector<uint8_t> rgba;
    rgba.resize((size_t)w * (size_t)h * 4);

    if (image.component == 4) {
        std::memcpy(rgba.data(), image.image.data(), rgba.size());
    }
    else if (image.component == 3) {
        // RGB -> RGBA
        const uint8_t* src = image.image.data();
        for (int i = 0; i < w * h; ++i) {
            rgba[i * 4 + 0] = src[i * 3 + 0];
            rgba[i * 4 + 1] = src[i * 3 + 1];
            rgba[i * 4 + 2] = src[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
        }
    }
    else if (image.component == 1) {
        const uint8_t* src = image.image.data();
        for (int i = 0; i < w * h; ++i) {
            rgba[i * 4 + 0] = src[i];
            rgba[i * 4 + 1] = src[i];
            rgba[i * 4 + 2] = src[i];
            rgba[i * 4 + 3] = 255;
        }
    }
    else {
        throw std::runtime_error("createTextureFromImage: unsupported component count");
    }

    // staging
    auto staging = Buffer::staging(context, (unsigned long)rgba.size());
    staging->upload(rgba.data(), (uint32_t)rgba.size(), 0);

	VkFormat format = srgb
        ? VK_FORMAT_R8G8B8A8_SRGB
        : VK_FORMAT_R8G8B8A8_UNORM;

    // image (GPU)
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = format;
    imgInfo.extent.width = (uint32_t)w;
    imgInfo.extent.height = (uint32_t)h;
    imgInfo.extent.depth = 1;
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage vkImage = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    if (vmaCreateImage(context->allocator, &imgInfo, &allocCI, &vkImage, &allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("createTextureFromImage: vmaCreateImage failed");

    // copy + layout transitions
    VkCommandBuffer cmd = context->beginOneTimeCommandBuffer();

    cmdTransitionImage(
        cmd, vkImage,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };

    vkCmdCopyBufferToImage(cmd, staging->buffer, vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    cmdTransitionImage(
        cmd, vkImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT);

    context->endOneTimeCommandBuffer(cmd, VulkanContext::Queue::GRAPHICS);

    // image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = vkImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView vkView = VK_NULL_HANDLE;
    if (vkCreateImageView(context->device, &viewInfo, nullptr, &vkView) != VK_SUCCESS)
        throw std::runtime_error("createTextureFromImage: vkCreateImageView failed");

    // sampler
    VkSamplerCreateInfo samp{};
    samp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp.magFilter = VK_FILTER_LINEAR;
    samp.minFilter = VK_FILTER_LINEAR;
    samp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samp.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp.maxAnisotropy = 1.0f;
    samp.anisotropyEnable = VK_FALSE;
    samp.minLod = 0.0f;
    samp.maxLod = 0.0f;

    VkSampler vkSampler = VK_NULL_HANDLE;
    if (vkCreateSampler(context->device, &samp, nullptr, &vkSampler) != VK_SUCCESS)
        throw std::runtime_error("createTextureFromImage: vkCreateSampler failed");

    GltfTextureGPU out{};
    out.image = vkImage;
    out.view = vkView;
    out.sampler = vkSampler;
    out.format = format;
    out.width = (uint32_t)w;
    out.height = (uint32_t)h;
    out.allocation = allocation;
    return out;
}

void GltfLoaderVulkan::destroyTexture(GltfTextureGPU& t)
{
    if (!context || context->device == VK_NULL_HANDLE) return;

    if (t.sampler) vkDestroySampler(context->device, t.sampler, nullptr);
    if (t.view)    vkDestroyImageView(context->device, t.view, nullptr);

    if (t.image && t.allocation) {
        vmaDestroyImage(context->allocator, t.image, (VmaAllocation)t.allocation);
    }

    t = {};
}

int GltfLoaderVulkan::getOrCreateTexture(int imageIndex, bool srgb)
{
    TextureKey key{ imageIndex, srgb };

    auto it = textureCache.find(key);
    if (it != textureCache.end()) {
        return it->second;
    }

    const auto& img = gltfModel.images[imageIndex];
    GltfTextureGPU tex = createTextureFromImage(img, srgb);

    int gpuIndex = (int)textures.size();
    textures.push_back(tex);
    textureCache[key] = gpuIndex;

    return gpuIndex;
}

bool GltfLoaderVulkan::loadModel(const std::string& path)
{
    // reset
    destroy();
    textures.clear();
    meshes.clear();
    drawItems.clear();
    gltfMeshIndexToGpuMeshes.clear();
    gltfModel = tinygltf::Model{};

    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok = false;
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".glb") {
        ok = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, path);
    }
    else {
        ok = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, path);
    }

    if (!warn.empty()) std::cerr << "[gltf warn] " << warn << "\n";
    if (!err.empty())  std::cerr << "[gltf err ] " << err << "\n";
    if (!ok) return false;

	textures.clear();

    for (int meshIndex = 0; meshIndex < (int)gltfModel.meshes.size(); ++meshIndex) {
        const auto& m = gltfModel.meshes[meshIndex];

        std::vector<uint32_t> gpuIndices;

        for (int primIndex = 0; primIndex < (int)m.primitives.size(); ++primIndex) {
            const auto& prim = m.primitives[primIndex];

            if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            // attributes
            auto attrs = prim.attributes;
            auto itPos = attrs.find("POSITION");
            auto itNor = attrs.find("NORMAL");
            auto itUv0 = attrs.find("TEXCOORD_0");
            auto itUv1 = attrs.find("TEXCOORD_1");

            if (itPos == attrs.end())
                throw std::runtime_error("Primitive missing POSITION");

            const bool hasNormal = (itNor != attrs.end());
            const bool hasUV0 = (itUv0 != attrs.end());
            const bool hasUV1 = (itUv1 != attrs.end());

            const tinygltf::Accessor& posAcc = gltfModel.accessors[itPos->second];
            if (posAcc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || posAcc.type != TINYGLTF_TYPE_VEC3)
                throw std::runtime_error("POSITION must be float3");

            // POSITION -> posCPU
            const uint8_t* posBase = accessorDataPtr(gltfModel, posAcc);
            const size_t   posStride = accessorStride(gltfModel, posAcc);
            const size_t   vtxCount = (size_t)posAcc.count;

            std::vector<glm::vec3> posCPU(vtxCount);
            for (size_t i = 0; i < vtxCount; ++i) {
                const float* p = reinterpret_cast<const float*>(posBase + i * posStride);
                posCPU[i] = glm::vec3(p[0], p[1], p[2]);
            }

            // indices
            if (prim.indices < 0) throw std::runtime_error("Primitive has no indices");
            const tinygltf::Accessor& idxAcc = gltfModel.accessors[prim.indices];
            const uint8_t* idxBase = accessorDataPtr(gltfModel, idxAcc);

            std::vector<uint32_t> idx32;
            std::vector<uint16_t> idx16;
            std::vector<uint32_t> idxAll32;

            VkIndexType indexType = VK_INDEX_TYPE_UINT16;

            if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                idx16.resize((size_t)idxAcc.count);
                const uint16_t* src = reinterpret_cast<const uint16_t*>(idxBase);
                for (size_t i = 0; i < (size_t)idxAcc.count; ++i) {
                    idx16[i] = src[i];
                    idxAll32.push_back((uint32_t)src[i]);
                }
                indexType = VK_INDEX_TYPE_UINT16;
            }
            else if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                idx32.resize((size_t)idxAcc.count);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(idxBase);
                for (size_t i = 0; i < (size_t)idxAcc.count; ++i) {
                    idx32[i] = src[i];
                }
                idxAll32 = idx32;
                indexType = VK_INDEX_TYPE_UINT32;
            }
            else {
                throw std::runtime_error("Index componentType must be UNSIGNED_SHORT or UNSIGNED_INT");
            }

            // NORMAL -> norCPU (read or generate)
            std::vector<glm::vec3> norCPU(vtxCount, glm::vec3(0.0f));
            if (hasNormal) {
                const tinygltf::Accessor& norAcc = gltfModel.accessors[itNor->second];
                if (norAcc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || norAcc.type != TINYGLTF_TYPE_VEC3)
                    throw std::runtime_error("NORMAL must be float3");

                const uint8_t* norBase = accessorDataPtr(gltfModel, norAcc);
                const size_t norStride = accessorStride(gltfModel, norAcc);

                for (size_t i = 0; i < vtxCount; ++i) {
                    const float* n = reinterpret_cast<const float*>(norBase + i * norStride);
                    glm::vec3 nn(n[0], n[1], n[2]);
                    float len2 = glm::dot(nn, nn);
                    norCPU[i] = (len2 > 1e-20f) ? (nn / std::sqrt(len2)) : glm::vec3(0, 1, 0);
                }
            }
            else {
                // generate smooth normals from triangles
                for (size_t i = 0; i + 2 < idxAll32.size(); i += 3) {
                    uint32_t i0 = idxAll32[i + 0];
                    uint32_t i1 = idxAll32[i + 1];
                    uint32_t i2 = idxAll32[i + 2];
                    if (i0 >= vtxCount || i1 >= vtxCount || i2 >= vtxCount) continue;

                    glm::vec3 e1 = posCPU[i1] - posCPU[i0];
                    glm::vec3 e2 = posCPU[i2] - posCPU[i0];
                    glm::vec3 fn = glm::cross(e1, e2);
                    float len2 = glm::dot(fn, fn);
                    if (len2 > 1e-12f) {
                        fn /= std::sqrt(len2);
                        norCPU[i0] += fn;
                        norCPU[i1] += fn;
                        norCPU[i2] += fn;
                    }
                }
                for (auto& n : norCPU) {
                    float len2 = glm::dot(n, n);
                    n = (len2 > 1e-20f) ? (n / std::sqrt(len2)) : glm::vec3(0, 1, 0);
                }
            }

            // UV0 -> uvCPU (read or XZ planar projection)
            std::vector<glm::vec2> uv0CPU(vtxCount, glm::vec2(0.0f));
            std::vector<glm::vec2> uv1CPU(vtxCount, glm::vec2(0.0f));
            if (hasUV0) {
                const tinygltf::Accessor& uvAcc = gltfModel.accessors[itUv0->second];
                if (uvAcc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || uvAcc.type != TINYGLTF_TYPE_VEC2)
                    throw std::runtime_error("TEXCOORD_0 must be float2");

                const uint8_t* uvBase = accessorDataPtr(gltfModel, uvAcc);
                const size_t uvStride = accessorStride(gltfModel, uvAcc);

                for (size_t i = 0; i < vtxCount; ++i) {
                    const float* t = reinterpret_cast<const float*>(uvBase + i * uvStride);
                    uv0CPU[i] = glm::vec2(t[0], t[1]);
                }
            }
            else {
                glm::vec3 bbMin(FLT_MAX), bbMax(-FLT_MAX);
                for (const auto& p : posCPU) {
                    bbMin = glm::min(bbMin, p);
                    bbMax = glm::max(bbMax, p);
                }
                glm::vec2 extent(bbMax.x - bbMin.x, bbMax.z - bbMin.z);
                extent = glm::max(extent, glm::vec2(1e-5f));

                for (size_t i = 0; i < vtxCount; ++i) {
                    const glm::vec3& p = posCPU[i];
                    uv0CPU[i] = glm::vec2(
                        (p.x - bbMin.x) / extent.x,
                        (p.z - bbMin.z) / extent.y
                    );
                }
            }

            if (hasUV1) {
                const auto& acc = gltfModel.accessors[itUv1->second];
                const uint8_t* base = accessorDataPtr(gltfModel, acc);
                size_t stride = accessorStride(gltfModel, acc);
                for (size_t i = 0; i < vtxCount; ++i) {
                    const float* t = reinterpret_cast<const float*>(base + i * stride);
                    uv1CPU[i] = { t[0], t[1] };
                }
            }

            // tangent optional (keep your logic)
            const uint8_t* tanBase = nullptr;
            size_t tanStride = 0;
            bool hasTangentAttr = false;
            auto itTan = attrs.find("TANGENT");
            tinygltf::Accessor tanAcc;
            if (itTan != attrs.end()) {
                tanAcc = gltfModel.accessors[itTan->second];
                if (tanAcc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && tanAcc.type == TINYGLTF_TYPE_VEC4) {
                    tanBase = accessorDataPtr(gltfModel, tanAcc);
                    tanStride = accessorStride(gltfModel, tanAcc);
                    hasTangentAttr = true;
                }
            }

            // build interleaved float array: pos3 + nrm3 + uv2 + tan4 = 12 floats
            uint32_t baseColorUvSet = 0; // default UV0
            if (prim.material >= 0 && prim.material < (int)gltfModel.materials.size()) {
                const auto& mat = gltfModel.materials[prim.material];
                auto itTex = mat.values.find("baseColorTexture");
                if (itTex != mat.values.end()) {
                    baseColorUvSet = (uint32_t)itTex->second.TextureTexCoord();
                }
            }

            std::vector<float> interleaved;
            interleaved.reserve(vtxCount * 12);

            for (size_t i = 0; i < vtxCount; ++i) {
                const glm::vec3& p = posCPU[i];
                const glm::vec3& n = norCPU[i];
                
                glm::vec2& uv = uv0CPU[i];
                if (baseColorUvSet == 1 && hasUV1) {
                    uv = uv1CPU[i];
				}

                interleaved.push_back(p.x); interleaved.push_back(p.y); interleaved.push_back(p.z);
                interleaved.push_back(n.x); interleaved.push_back(n.y); interleaved.push_back(n.z);
                interleaved.push_back(uv.x); interleaved.push_back(uv.y);

                if (hasTangentAttr && tanBase) {
                    const float* tg = reinterpret_cast<const float*>(tanBase + i * tanStride);
                    interleaved.push_back(tg[0]); interleaved.push_back(tg[1]);
                    interleaved.push_back(tg[2]); interleaved.push_back(tg[3]);
                }
                else {
                    interleaved.push_back(1.0f); interleaved.push_back(0.0f);
                    interleaved.push_back(0.0f); interleaved.push_back(1.0f);
                }
            }

            // upload vertex/index to GPU using staging (keep your baseline path)
            auto vb = Buffer::vertex(context, (uint64_t)interleaved.size() * sizeof(float), false, "GLTF_VBO");
            auto vbStaging = Buffer::staging(context, (unsigned long)interleaved.size() * sizeof(float));
            vbStaging->upload(interleaved.data(), (uint32_t)interleaved.size() * sizeof(float), 0);
            vb->uploadFrom(vbStaging);

            std::shared_ptr<Buffer> ib;
            if (indexType == VK_INDEX_TYPE_UINT16) {
                ib = Buffer::index(context, (uint64_t)idx16.size() * sizeof(uint16_t), false, "GLTF_IBO");
                auto ibStaging = Buffer::staging(context, (unsigned long)idx16.size() * sizeof(uint16_t));
                ibStaging->upload(idx16.data(), (uint32_t)idx16.size() * sizeof(uint16_t), 0);
                ib->uploadFrom(ibStaging);
            }
            else {
                ib = Buffer::index(context, (uint64_t)idx32.size() * sizeof(uint32_t), false, "GLTF_IBO");
                auto ibStaging = Buffer::staging(context, (unsigned long)idx32.size() * sizeof(uint32_t));
                ibStaging->upload(idx32.data(), (uint32_t)idx32.size() * sizeof(uint32_t), 0);
                ib->uploadFrom(ibStaging);
            }

            // material (KEEP your baseline behavior so textures show up)
            GltfMeshGPU gpu{};
            gpu.vertexBuffer = vb;
            gpu.indexBuffer = ib;
            gpu.indexCount = (uint32_t)idxAcc.count;
            gpu.indexType = indexType;
            gpu.model = glm::mat4(1.0f);

            if (prim.material >= 0 && prim.material < (int)gltfModel.materials.size()) {
                const auto& mat = gltfModel.materials[prim.material];

                // baseColorFactor
                auto it = mat.values.find("baseColorFactor");
                if (it != mat.values.end() && it->second.number_array.size() == 4) {
                    gpu.baseColorFactor = glm::vec4(
                        (float)it->second.number_array[0],
                        (float)it->second.number_array[1],
                        (float)it->second.number_array[2],
                        (float)it->second.number_array[3]);
                }

                // baseColorTexture (SRGB)
                auto itTex = mat.values.find("baseColorTexture");
                if (itTex != mat.values.end()) {
                    int texIndex = itTex->second.TextureIndex();
                    if (texIndex >= 0 && texIndex < (int)gltfModel.textures.size()) {
                        int imgIndex = gltfModel.textures[texIndex].source;
                        if (imgIndex >= 0 && imgIndex < (int)gltfModel.images.size()) {
                            gpu.hasBaseColorTex = true;
                            gpu.baseColorTex = getOrCreateTexture(imgIndex, /*srgb=*/true);
                            gpu.uvSetBaseColor = (uint8_t)itTex->second.TextureTexCoord();
                        }
                    }
                }

                // metallicFactor / roughnessFactor
                auto itMR = mat.values.find("metallicFactor");
                if (itMR != mat.values.end())
                    gpu.metallicFactor = (float)itMR->second.number_value;

                auto itRF = mat.values.find("roughnessFactor");
                if (itRF != mat.values.end())
                    gpu.roughnessFactor = (float)itRF->second.number_value;

                // metallicRoughnessTexture (UNORM)
                auto itMRTex = mat.values.find("metallicRoughnessTexture");
                if (itMRTex != mat.values.end()) {
                    int texIndex = itMRTex->second.TextureIndex();
                    if (texIndex >= 0 && texIndex < (int)gltfModel.textures.size()) {
                        int imgIndex = gltfModel.textures[texIndex].source;
                        if (imgIndex >= 0 && imgIndex < (int)gltfModel.images.size()) {
                            gpu.hasMetallicRoughnessTex = true;
                            gpu.metallicRoughnessTex = getOrCreateTexture(imgIndex, /*srgb=*/false);
                            gpu.uvSetMR = (uint8_t)itMRTex->second.TextureTexCoord();
                        }
                    }
                }

                // normalTexture (UNORM)
                auto itNorm = mat.additionalValues.find("normalTexture");
                if (itNorm != mat.additionalValues.end()) {
                    int texIndex = itNorm->second.TextureIndex();
                    if (texIndex >= 0 && texIndex < (int)gltfModel.textures.size()) {
                        int imgIndex = gltfModel.textures[texIndex].source;
                        if (imgIndex >= 0 && imgIndex < (int)gltfModel.images.size()) {
                            gpu.hasNormalTex = true;
                            gpu.normalTex = getOrCreateTexture(imgIndex, /*srgb=*/false);
                            gpu.normalScale = (float)itNorm->second.TextureScale(); // 比 number_value 更稳
                            gpu.hasTangents = true;
                            gpu.uvSetNormal = (uint8_t)itNorm->second.TextureTexCoord();
                        }
                    }
                }

                // occlusionTexture (UNORM)
                auto itOcc = mat.additionalValues.find("occlusionTexture");
                if (itOcc != mat.additionalValues.end()) {
                    int texIndex = itOcc->second.TextureIndex();
                    if (texIndex >= 0 && texIndex < (int)gltfModel.textures.size()) {
                        int imgIndex = gltfModel.textures[texIndex].source;
                        if (imgIndex >= 0 && imgIndex < (int)gltfModel.images.size()) {
                            gpu.hasOcclusionTex = true;
                            gpu.occlusionTex = getOrCreateTexture(imgIndex, /*srgb=*/false);
                            gpu.occlusionStrength = (float)itOcc->second.TextureStrength();
                            gpu.uvSetOcclusion = (uint8_t)itOcc->second.TextureTexCoord();
                        }
                    }
                }

                // emissiveFactor
                auto itEmF = mat.additionalValues.find("emissiveFactor");
                if (itEmF != mat.additionalValues.end() && itEmF->second.number_array.size() == 3) {
                    gpu.emissiveFactor = glm::vec3(
                        (float)itEmF->second.number_array[0],
                        (float)itEmF->second.number_array[1],
                        (float)itEmF->second.number_array[2]);
                }

                // emissiveTexture (SRGB)
                auto itEmT = mat.additionalValues.find("emissiveTexture");
                if (itEmT != mat.additionalValues.end()) {
                    int texIndex = itEmT->second.TextureIndex();
                    if (texIndex >= 0 && texIndex < (int)gltfModel.textures.size()) {
                        int imgIndex = gltfModel.textures[texIndex].source;
                        if (imgIndex >= 0 && imgIndex < (int)gltfModel.images.size()) {
                            gpu.hasEmissiveTex = true;
                            gpu.emissiveTex = getOrCreateTexture(imgIndex, /*srgb=*/true);
                            gpu.uvSetEmissive = (uint8_t)itEmT->second.TextureTexCoord();
                        }
                    }
                }

                // KHR_materials_sheen
                auto itSheenExt = mat.extensions.find("KHR_materials_sheen");
                if (itSheenExt != mat.extensions.end()) {

                    const tinygltf::Value& sheen = itSheenExt->second;
                    if (!sheen.IsObject()) {
                        // malformed, ignore
                    }
                    else {
                        gpu.hasSheen = true;
                        gpu.materialFlags |= MATERIAL_FLAG_SHEEN;
                        gpu.materialFlags |= MATERIAL_FLAG_CLOTH;
                        gpu.materialFlags &= ~MATERIAL_FLAG_FOLIAGE;

                        // ---- sheenColorFactor (vec3) ----
                        if (sheen.Has("sheenColorFactor")) {
                            const auto& arr = sheen.Get("sheenColorFactor");
                            if (arr.IsArray() && arr.ArrayLen() >= 3) {
                                gpu.sheenColorFactor = glm::vec3(
                                    (float)arr.Get(0).GetNumberAsDouble(),
                                    (float)arr.Get(1).GetNumberAsDouble(),
                                    (float)arr.Get(2).GetNumberAsDouble()
                                );
                            }
                        }

                        // ---- sheenRoughnessFactor (float) ----
                        if (sheen.Has("sheenRoughnessFactor")) {
                            gpu.sheenRoughnessFactor =
                                (float)sheen.Get("sheenRoughnessFactor").GetNumberAsDouble();
                        }

                        // ---- sheenColorTexture ----
                        if (sheen.Has("sheenColorTexture")) {
                            const auto& texInfo = sheen.Get("sheenColorTexture");
                            if (texInfo.IsObject()) {

                                int texIndex = -1;
                                int texCoord = 0;

                                if (texInfo.Has("index")) {
                                    texIndex = texInfo.Get("index").GetNumberAsInt();
                                }
                                if (texInfo.Has("texCoord")) {
                                    texCoord = texInfo.Get("texCoord").GetNumberAsInt();
                                }

                                if (texIndex >= 0 &&
                                    texIndex < (int)gltfModel.textures.size()) {

                                    int imgIndex =
                                        gltfModel.textures[texIndex].source;

                                    if (imgIndex >= 0 &&
                                        imgIndex < (int)gltfModel.images.size()) {

                                        gpu.sheenColorTex =
                                            getOrCreateTexture(imgIndex, /*srgb=*/true);

                                        gpu.uvSetSheenColor = (uint8_t)texCoord;
                                    }
                                }
                            }
                        }

                        // ---- sheenRoughnessTexture (linear) ----
                        if (sheen.Has("sheenRoughnessTexture")) {
                            const auto& texInfo = sheen.Get("sheenRoughnessTexture");
                            if (texInfo.IsObject()) {
                                int texIndex = -1;
                                int texCoord = 0;

                                if (texInfo.Has("index")) {
                                    texIndex = texInfo.Get("index").GetNumberAsInt();
                                }
                                if (texInfo.Has("texCoord")) {
                                    texCoord = texInfo.Get("texCoord").GetNumberAsInt();
                                }

                                if (texIndex >= 0 && texIndex < (int)gltfModel.textures.size()) {
                                    int imgIndex = gltfModel.textures[texIndex].source;
                                    if (imgIndex >= 0 && imgIndex < (int)gltfModel.images.size()) {
                                        // sheenRoughnessTexture is linear
                                        gpu.sheenRoughnessTex = getOrCreateTexture(imgIndex, /*srgb=*/false);
                                        gpu.uvSetSheenRoughness = (uint8_t)texCoord;
                                    }
                                }
                            }
                        }
                    }
                }

                // KHR_materials_transmission
                auto itTrExt = mat.extensions.find("KHR_materials_transmission");
                if (itTrExt != mat.extensions.end()) {
                    const tinygltf::Value& tr = itTrExt->second;
                    if (tr.IsObject()) {
                        gpu.hasTransmission = true;
                        gpu.materialFlags |= MATERIAL_FLAG_TRANSMISSION;

                        // transmissionFactor (default 0)
                        if (tr.Has("transmissionFactor")) {
                            gpu.transmissionFactor =
                                (float)tr.Get("transmissionFactor").GetNumberAsDouble();
                        }

                        // transmissionTexture
                        if (tr.Has("transmissionTexture")) {
                            const auto& texInfo = tr.Get("transmissionTexture");
                            if (texInfo.IsObject()) {
                                int texIndex = texInfo.Has("index") ? texInfo.Get("index").GetNumberAsInt() : -1;
                                int texCoord = texInfo.Has("texCoord") ? texInfo.Get("texCoord").GetNumberAsInt() : 0;

                                if (texIndex >= 0 && texIndex < (int)gltfModel.textures.size()) {
                                    int imgIndex = gltfModel.textures[texIndex].source;
                                    if (imgIndex >= 0 && imgIndex < (int)gltfModel.images.size()) {
                                        gpu.transmissionTex = getOrCreateTexture(imgIndex, /*srgb=*/false); // transmission是线性
                                        gpu.uvSetTransmission = (uint8_t)texCoord;
                                    }
                                }
                            }
                        }
                    }
                }


                if (mat.alphaMode == "MASK" || mat.alphaMode == "BLEND") {
                    gpu.materialFlags |= MATERIAL_FLAG_FOLIAGE;
                    gpu.materialFlags |= MATERIAL_FLAG_THIN;
                }
                // cloth implies thin
                if (gpu.materialFlags & MATERIAL_FLAG_CLOTH) {
                    gpu.materialFlags |= MATERIAL_FLAG_THIN;
                }

                // sheen and foliage should not mix
                if (gpu.materialFlags & MATERIAL_FLAG_SHEEN) {
                    gpu.materialFlags &= ~MATERIAL_FLAG_FOLIAGE;
                }
            }


            uint32_t gpuIndex = (uint32_t)meshes.size();
            meshes.push_back(std::move(gpu));
            gpuIndices.push_back(gpuIndex);
        }

        gltfMeshIndexToGpuMeshes[meshIndex] = std::move(gpuIndices);
    }

    buildDrawItemsFromScene();

    size_t cntBaseTex = 0, cntNormalTex = 0, cntMRTex = 0;
    for (auto& m : meshes) {
        cntBaseTex += m.hasBaseColorTex;
        cntNormalTex += m.hasNormalTex;
        cntMRTex += m.hasMetallicRoughnessTex;
    }

    std::cout << "[GLTF] meshes={} baseTex={} normalTex={} mrTex={}" <<
        meshes.size() << " " << cntBaseTex << " " << cntNormalTex << " " << cntMRTex << std::endl;

    return true;
}

void GltfLoaderVulkan::buildDrawItemsFromScene()
{
    drawItems.clear();

    if (gltfModel.scenes.empty()) return;
    int sceneIndex = (gltfModel.defaultScene >= 0)
        ? gltfModel.defaultScene
        : 0;
    const auto& sc = gltfModel.scenes[sceneIndex];

    glm::mat4 rootWorld = makeRootMatrix(rootXform);

    uint32_t nextDrawId = 1;

    std::function<void(int, const glm::mat4&)> dfs = [&](int nodeIndex, const glm::mat4& parentWorld) {
        const auto& n = gltfModel.nodes[nodeIndex];
        glm::mat4 local = nodeLocalMatrix(n);
        glm::mat4 world = parentWorld * local;

        // node references a mesh
        if (n.mesh >= 0) {
            auto it = gltfMeshIndexToGpuMeshes.find(n.mesh);
            if (it != gltfMeshIndexToGpuMeshes.end()) {
                for (uint32_t gpuMeshIndex : it->second) {
                    GltfDrawItem di{};
                    di.meshGpuIndex = gpuMeshIndex;
                    di.world = world;
                    di.drawId = nextDrawId++;
                    drawItems.push_back(di);
                }
            }
        }

        for (int c : n.children) dfs(c, world);
    };

    for (int root : sc.nodes) dfs(root, rootWorld);
}

void GltfLoaderVulkan::destroy()
{
    // textures
    for (auto& t : textures) {
        destroyTexture(t);
    }
    textures.clear();

    // meshes buffers are owned by shared_ptr<Buffer> and will release automatically
    meshes.clear();
    drawItems.clear();
    gltfMeshIndexToGpuMeshes.clear();

    gltfModel = tinygltf::Model{};
}

void GltfLoaderVulkan::dumpSummary() const
{
    std::cout << "==== GLTF Vulkan Loader Summary ====\n";
    std::cout << "Meshes(GPU): " << meshes.size() << "\n";
    std::cout << "Textures(GPU): " << textures.size() << "\n";
    std::cout << "DrawItems: " << drawItems.size() << "\n";
    std::cout << "Scenes: " << gltfModel.scenes.size() << ", Nodes: " << gltfModel.nodes.size()
        << ", Materials: " << gltfModel.materials.size() << "\n";
    if (!drawItems.empty()) {
        std::cout << "First draw item meshGpuIndex=" << drawItems[0].meshGpuIndex << "\n";
    }
}
