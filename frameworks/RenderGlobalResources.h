#pragma once

#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// forward declarations
class VulkanContext;
class Swapchain;
class Buffer;
class QueryManager;
class Camera;

struct alignas(16) UniformBuffer {
    glm::vec4 camera_position;

    glm::mat4 view_mat;
    glm::mat4 proj_mat;
    glm::mat4 view_proj;
    uint32_t  width;
    uint32_t  height;
    float     tan_fovx;
    float     tan_fovy;
    glm::vec4 gaze_params; // x,y in pixels; z=R0; w=R1
};

struct GazeData
{
    // x,y: pixel space
    // z:   R0
    // w:   R1
    glm::vec4 gazeParams = glm::vec4(0.0f);

	// used for smoothing
    // glm::vec2 velocityPx;
    // float     timestamp;
};

struct RenderGlobalResources
{
    std::shared_ptr<VulkanContext> context;

    std::shared_ptr<Swapchain>     swapchain;

    std::shared_ptr<Buffer>        uniformBuffer;
    std::shared_ptr<Buffer>        lightUboBuffer;

	bool 						enableQueryManager = false;
    std::shared_ptr<QueryManager>  queryManager;

    Camera* camera = nullptr;

    std::shared_ptr<GazeData>      gazeData;
};
