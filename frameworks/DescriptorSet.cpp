#include "DescriptorSet.h"

DescriptorSet::DescriptorSet(const std::shared_ptr<VulkanContext>& ctx, uint8_t frames)
    : context(ctx), framesInFlight(frames)
{
    if (!context) {
        throw std::runtime_error("DescriptorSet: context is null");
    }
    if (framesInFlight == 0) {
        throw std::runtime_error("DescriptorSet: framesInFlight must be >= 1");
    }
}

DescriptorSet::~DescriptorSet()
{
    if (!context) return;
    VkDevice device = context->device;

    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    descriptorSets.clear();
    bindings.clear();
    maxOptions = 1;
}

void DescriptorSet::bindBufferToDescriptorSet(uint32_t binding,
    VkDescriptorType type,
    VkShaderStageFlags stage,
    std::shared_ptr<Buffer> buffer)
{
    if (!buffer) {
        throw std::runtime_error("bindBufferToDescriptorSet: buffer is null");
    }

    VkDescriptorSetLayoutBinding layout{};
    layout.binding = binding;
    layout.descriptorCount = 1;
    layout.descriptorType = type;
    layout.stageFlags = stage;
    layout.pImmutableSamplers = nullptr;

    auto& vec = bindings[binding];

    if (!vec.empty()) {
        const auto& last = vec.back().layoutBinding;
        if (last.binding != layout.binding ||
            last.descriptorType != layout.descriptorType ||
            last.descriptorCount != layout.descriptorCount ||
            last.stageFlags != layout.stageFlags)
        {
            throw std::runtime_error("bindBufferToDescriptorSet: binding already exists with different layout");
        }
    }

    DescriptorBinding b{};
    b.type = type;
    b.layoutBinding = layout;
    b.buffer = buffer;

    b.bufferInfo.buffer = buffer->buffer;
    b.bufferInfo.offset = 0;
    b.bufferInfo.range = buffer->size;

    vec.push_back(b);
}

void DescriptorSet::bindImageToDescriptorSet(uint32_t binding,
    VkDescriptorType type,
    VkShaderStageFlags stage,
    std::shared_ptr<Image> image)
{
    if (!image) {
        throw std::runtime_error("bindImageToDescriptorSet: image is null");
    }

    VkDescriptorSetLayoutBinding layout{};
    layout.binding = binding;
    layout.descriptorCount = 1;
    layout.descriptorType = type;
    layout.stageFlags = stage;
    layout.pImmutableSamplers = nullptr;

    auto& vec = bindings[binding];

    if (!vec.empty()) {
        const auto& last = vec.back().layoutBinding;
        if (last.binding != layout.binding ||
            last.descriptorType != layout.descriptorType ||
            last.descriptorCount != layout.descriptorCount ||
            last.stageFlags != layout.stageFlags)
        {
            throw std::runtime_error("bindImageToDescriptorSet: binding already exists with different layout");
        }
    }

    DescriptorBinding b{};
    b.type = type;
    b.layoutBinding = layout;
    b.image = image;

    b.imageInfo.sampler = VK_NULL_HANDLE;
    b.imageInfo.imageView = image->imageView;
    b.imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    vec.push_back(b);
}

void DescriptorSet::bindCombinedImageSamplerToDescriptorSet(
    uint32_t binding,
    VkShaderStageFlags stageFlags,
    std::shared_ptr<Image> image,
    VkSampler sampler,
    VkImageLayout imageLayout)
{
    if (!image) {
        throw std::runtime_error("bindCombinedImageSamplerToDescriptorSet: image is null");
    }
    if (sampler == VK_NULL_HANDLE) {
        throw std::runtime_error("bindCombinedImageSamplerToDescriptorSet: sampler is null");
    }
    if (image->imageView == VK_NULL_HANDLE) {
        throw std::runtime_error("bindCombinedImageSamplerToDescriptorSet: imageView is null");
    }

    VkDescriptorSetLayoutBinding layout{};
    layout.binding = binding;
    layout.descriptorCount = 1;
    layout.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layout.stageFlags = stageFlags;
    layout.pImmutableSamplers = nullptr;

    auto& vec = bindings[binding];

    if (!vec.empty()) {
        const auto& last = vec.back().layoutBinding;
        if (last.binding != layout.binding ||
            last.descriptorType != layout.descriptorType ||
            last.descriptorCount != layout.descriptorCount ||
            last.stageFlags != layout.stageFlags)
        {
            throw std::runtime_error("bindCombinedImageSamplerToDescriptorSet: binding already exists with different layout");
        }
    }

    DescriptorBinding b{};
    b.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.layoutBinding = layout;
    b.image = image;

    b.imageInfo.sampler = sampler;
    b.imageInfo.imageView = image->imageView;
    b.imageInfo.imageLayout = imageLayout;

    vec.push_back(b);
}

void DescriptorSet::bindCombinedImageSamplerArrayToDescriptorSet(
    uint32_t binding,
    VkShaderStageFlags stageFlags,
    const std::vector<std::shared_ptr<Image>>& images,
    const std::vector<VkSampler>& samplers,
    VkImageLayout imageLayout)
{
    if (images.empty()) {
        throw std::runtime_error("bindCombinedImageSamplerArrayToDescriptorSet: images is empty");
    }
    if (images.size() != samplers.size()) {
        throw std::runtime_error("bindCombinedImageSamplerArrayToDescriptorSet: images.size != samplers.size");
    }

    for (size_t i = 0; i < images.size(); ++i) {
        if (!images[i]) {
            throw std::runtime_error("bindCombinedImageSamplerArrayToDescriptorSet: images[i] is null");
        }
        if (samplers[i] == VK_NULL_HANDLE) {
            throw std::runtime_error("bindCombinedImageSamplerArrayToDescriptorSet: samplers[i] is null");
        }
        if (images[i]->imageView == VK_NULL_HANDLE) {
            throw std::runtime_error("bindCombinedImageSamplerArrayToDescriptorSet: images[i]->imageView is null");
        }
    }

    VkDescriptorSetLayoutBinding layout{};
    layout.binding = binding;
    layout.descriptorCount = static_cast<uint32_t>(images.size());
    layout.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layout.stageFlags = stageFlags;
    layout.pImmutableSamplers = nullptr;

    auto& vec = bindings[binding];

    if (!vec.empty()) {
        const auto& last = vec.back().layoutBinding;
        if (last.binding != layout.binding ||
            last.descriptorType != layout.descriptorType ||
            last.descriptorCount != layout.descriptorCount ||
            last.stageFlags != layout.stageFlags)
        {
            throw std::runtime_error("bindCombinedImageSamplerArrayToDescriptorSet: binding already exists with different layout");
        }
    }

    DescriptorBinding b{};
    b.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.layoutBinding = layout;

    b.images = images;

    b.imageInfos.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        b.imageInfos[i].sampler = samplers[i];
        b.imageInfos[i].imageView = images[i]->imageView;
        b.imageInfos[i].imageLayout = imageLayout;
    }

    vec.push_back(std::move(b));
}

void DescriptorSet::updateCombinedImageSampler(
    uint32_t binding,
    uint32_t frameIndex,
    const std::shared_ptr<Image>& image,
    VkSampler sampler,
    VkImageLayout layout)
{
    if (!image) throw std::runtime_error("DescriptorSet::updateCombinedImageSampler: image is null");

    if (frameIndex >= framesInFlight) {
        throw std::runtime_error("DescriptorSet::updateCombinedImageSampler: frameIndex out of range");
    }

    auto it = bindings.find(binding);
    if (it == bindings.end() || it->second.empty()) {
        throw std::runtime_error("DescriptorSet::updateCombinedImageSampler: binding not found/empty");
    }

    DescriptorBinding& b = it->second[0]; // option 0

    b.image = image;
    b.imageInfo = {};
    b.imageInfo.imageLayout = layout;
    b.imageInfo.imageView = image->imageView;
    b.imageInfo.sampler = sampler;

    const uint32_t setIndex = frameIndex * static_cast<uint32_t>(maxOptions) + 0u;
    if (setIndex >= descriptorSets.size()) {
        throw std::runtime_error("DescriptorSet::updateCombinedImageSampler: internal setIndex out of range");
    }

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSets[setIndex];
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &b.imageInfo;

    vkUpdateDescriptorSets(context->device, 1, &write, 0, nullptr);
}


void DescriptorSet::bindCombinedImageSamplerRaw(
    uint32_t binding,
    VkShaderStageFlags stageFlags,
    VkImageView imageView,
    VkSampler sampler,
    VkImageLayout imageLayout)
{
    DescriptorBinding b{};
    b.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.layoutBinding.binding = binding;
    b.layoutBinding.descriptorCount = 1;
    b.layoutBinding.descriptorType = b.type;
    b.layoutBinding.stageFlags = stageFlags;

    b.rawImageView = imageView;
    b.rawSampler = sampler;
    b.rawLayout = imageLayout;

    b.rawImageInfo.imageView = imageView;
    b.rawImageInfo.sampler = sampler;
    b.rawImageInfo.imageLayout = imageLayout;

    bindings[binding] = { b };
}

void DescriptorSet::bindStorageImageToDescriptorSet(
    uint32_t binding,
    VkShaderStageFlags stageFlags,
    std::shared_ptr<Image> image,
    VkImageLayout imageLayout)
{
    if (!image) {
        throw std::runtime_error("bindStorageImageToDescriptorSet: image is null");
    }

    VkDescriptorSetLayoutBinding layout{};
    layout.binding = binding;
    layout.descriptorCount = 1;
    layout.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layout.stageFlags = stageFlags;
    layout.pImmutableSamplers = nullptr;

    auto& vec = bindings[binding];

    if (!vec.empty()) {
        const auto& last = vec.back().layoutBinding;
        if (last.binding != layout.binding ||
            last.descriptorType != layout.descriptorType ||
            last.descriptorCount != layout.descriptorCount ||
            last.stageFlags != layout.stageFlags)
        {
            throw std::runtime_error("bindStorageImageToDescriptorSet: binding already exists with different layout");
        }
    }

    DescriptorBinding b{};
    b.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.layoutBinding = layout;
    b.image = image;

    b.imageInfo.sampler = VK_NULL_HANDLE;
    b.imageInfo.imageView = image->imageView;
    b.imageInfo.imageLayout = imageLayout;

    vec.push_back(b);
}

void DescriptorSet::updateStorageImage(
    uint32_t binding,
    uint32_t frameIndex,
    const std::shared_ptr<Image>& image,
    VkImageLayout layout)
{
    if (!image) throw std::runtime_error("DescriptorSet::updateStorageImage: image is null");
    if (frameIndex >= framesInFlight) throw std::runtime_error("DescriptorSet::updateStorageImage: frameIndex out of range");

    auto it = bindings.find(binding);
    if (it == bindings.end() || it->second.empty()) {
        throw std::runtime_error("DescriptorSet::updateStorageImage: binding not found/empty");
    }

    DescriptorBinding& b = it->second[0]; // option 0
    b.image = image;
    b.imageInfo = {};
    b.imageInfo.imageView = image->imageView;
    b.imageInfo.imageLayout = layout;
    b.imageInfo.sampler = VK_NULL_HANDLE;

    const uint32_t setIndex = frameIndex * static_cast<uint32_t>(maxOptions) + 0u;
    if (setIndex >= descriptorSets.size()) {
        throw std::runtime_error("DescriptorSet::updateStorageImage: internal setIndex out of range");
    }

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSets[setIndex];
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &b.imageInfo;

    vkUpdateDescriptorSets(context->device, 1, &write, 0, nullptr);
}

void DescriptorSet::build()
{
    if (!context) {
        throw std::runtime_error("DescriptorSet::build: context is null");
    }
    if (context->descriptorPool == VK_NULL_HANDLE) {
        throw std::runtime_error("DescriptorSet::build: context->descriptorPool is null (did you create it in VulkanContext?)");
    }

    maxOptions = 1;

    std::vector<uint32_t> bindingKeys;
    bindingKeys.reserve(bindings.size());
    for (const auto& kv : bindings) bindingKeys.push_back(kv.first);
    std::sort(bindingKeys.begin(), bindingKeys.end());

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.reserve(bindingKeys.size());

    for (uint32_t key : bindingKeys) {
        const auto& opts = bindings.at(key);
        if (opts.empty()) continue;

        layoutBindings.push_back(opts[0].layoutBinding);

        if (opts.size() != 1) {
            if (maxOptions == 1) {
                maxOptions = opts.size();
            }
            else if (maxOptions != opts.size()) {
                throw std::runtime_error("DescriptorSet::build: inconsistent option count across bindings");
            }
        }
    }

    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context->device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    if (vkCreateDescriptorSetLayout(context->device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("DescriptorSet::build: failed to create descriptor set layout");
    }

    const uint32_t totalSets = static_cast<uint32_t>(framesInFlight) * static_cast<uint32_t>(maxOptions);

    descriptorSets.assign(totalSets, VK_NULL_HANDLE);

    std::vector<VkDescriptorSetLayout> layouts(totalSets, descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = context->descriptorPool; // IMPORTANT: use global pool (original behavior)
    allocInfo.descriptorSetCount = totalSets;
    allocInfo.pSetLayouts = layouts.data();

    VkResult ar = vkAllocateDescriptorSets(context->device, &allocInfo, descriptorSets.data());
    if (ar != VK_SUCCESS) {
        throw std::runtime_error("DescriptorSet::build: failed to allocate descriptor sets from context->descriptorPool");
    }

    for (uint32_t  frame = 0; frame < framesInFlight; ++frame) {
        for (uint32_t  option = 0; option < static_cast<uint32_t>(maxOptions); ++option) {

            const uint32_t setIndex = static_cast<uint32_t>(frame) * static_cast<uint32_t>(maxOptions) + option;
            VkDescriptorSet dstSet = descriptorSets[setIndex];

            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(bindingKeys.size());

            for (uint32_t binding : bindingKeys) {
                const auto& opts = bindings.at(binding);
                if (opts.empty()) continue;

                const DescriptorBinding* b = nullptr;
                if (opts.size() == 1) {
                    b = &opts[0];
                }
                else {
                    if (option >= opts.size()) {
                        throw std::runtime_error("DescriptorSet::build: option index out of range for a binding");
                    }
                    b = &opts[option];
                }

                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = dstSet;
                w.dstBinding = binding;
                w.dstArrayElement = 0;
                w.descriptorType = b->type;

                if (b->buffer) {
                    b->buffer->boundToDescriptorSet(
                        std::weak_ptr<DescriptorSet>(shared_from_this()),
                        setIndex,
                        binding,
                        b->type
                    );

					w.descriptorCount = 1;
                    w.pBufferInfo = &b->bufferInfo;
                }
                else {
                    if (!b->imageInfos.empty()) {
                        w.descriptorCount = static_cast<uint32_t>(b->imageInfos.size());
                        w.pImageInfo = b->imageInfos.data();
                    }
                    else if (b->rawImageView != VK_NULL_HANDLE) {
                        w.descriptorCount = 1;
                        w.pImageInfo = &b->rawImageInfo;
                    }
                    else if (b->image) {
                        w.descriptorCount = 1;
                        w.pImageInfo = &b->imageInfo;
                    }
                    else {
                        throw std::runtime_error("DescriptorSet::build: binding has neither buffer nor image");
                    }
                }

                writes.push_back(w);
            }

            if (!writes.empty()) {
                vkUpdateDescriptorSets(context->device,
                    static_cast<uint32_t>(writes.size()), writes.data(),
                    0, nullptr);
            }
        }
    }
}

VkDescriptorSet DescriptorSet::getDescriptorSet(uint8_t frame, uint32_t option) const
{
    if (frame >= framesInFlight) {
        throw std::runtime_error("DescriptorSet::getDescriptorSet: frame out of range");
    }
    if (option >= maxOptions) {
        throw std::runtime_error("DescriptorSet::getDescriptorSet: option out of range");
    }
    
    const size_t idx = static_cast<size_t>(frame) * maxOptions + static_cast<size_t>(option);
    if (idx >= descriptorSets.size()) {
        throw std::runtime_error("DescriptorSet::getDescriptorSet: internal index out of range");
    }
    return descriptorSets[idx];
}
