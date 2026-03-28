#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

struct Image {
    VkImage      image = VK_NULL_HANDLE;
	VkImageView  mip0View = VK_NULL_HANDLE; // optional, view for mip level 0
	VkImageView  imageView = VK_NULL_HANDLE; // main view for all mips/layers
    VkFormat     format = VK_FORMAT_UNDEFINED;
    VkExtent2D   extent = { 0, 0 };
    VkFramebuffer framebuffer = VK_NULL_HANDLE; // optional
    VmaAllocation allocation = VK_NULL_HANDLE;

    Image() = default;

    Image(VkImage img,
        VkImageView view,
        VkFormat fmt,
        VkExtent2D ext,
        VkFramebuffer fb = VK_NULL_HANDLE,
        VmaAllocation allc = VK_NULL_HANDLE)
        : image(img)
        , imageView(view)
        , format(fmt)
        , extent(ext)
        , framebuffer(fb)
        , allocation(allc){
    }
    Image(VkImage img,
        VkImageView mip0,
        VkImageView view,
        VkFormat fmt,
        VkExtent2D ext,
        VkFramebuffer fb = VK_NULL_HANDLE,
        VmaAllocation allc = VK_NULL_HANDLE)
        : image(img)
        , mip0View(mip0)
        , imageView(view)
        , format(fmt)
        , extent(ext)
        , framebuffer(fb)
        , allocation(allc) {
    }

};
