#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>

struct CameraSnapshot
{
    glm::vec3 position;
    glm::quat rotation;

    float fov_y_deg;
    float near_plane;
    float far_plane;

    float aspect; // width / height
};

class Camera
{
public:
    Camera() = default;

    Camera(const glm::vec3& pos,
        const glm::quat& rot,
        float             fovDeg,
        float             zNear,
        float             zFar)
        : position(pos),
        rotation(rot),
        fov(fovDeg),
        nearPlane(zNear),
        farPlane(zFar)
    {
    }

    void translate(const glm::vec3& delta)
    {
        translateLocal(delta);
    }
    void translateLocal(const glm::vec3& delta)
    {
        position += rotation * delta;
    }

    void translateWorld(const glm::vec3& delta)
    {
        position += delta;
    }

    glm::vec3 forward() const { return rotation * glm::vec3(0, 0, -1); }
    glm::vec3 right()   const { return rotation * glm::vec3(1, 0, 0); }
    glm::vec3 up()      const { return rotation * glm::vec3(0, 1, 0); }

    glm::mat4 viewMatrix() const
    {
        glm::quat q = glm::normalize(rotation);
        glm::mat4 Rinv = glm::mat4_cast(glm::conjugate(q));
        glm::mat4 Tinv = glm::translate(glm::mat4(1.0f), -position);
        return Rinv * Tinv;
    }
    glm::mat4 projectionMatrix(float aspect) const
    {
        return glm::perspectiveRH_ZO(
            glm::radians(fov),
            aspect,
            nearPlane,
            farPlane
        );
    }

public:
    glm::vec3 position{ 0,0,0 };
    glm::quat rotation{ 1,0,0,0 };

    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.f;
};

void appendCameraLog(
    const CameraSnapshot& snap,
    const std::string& filepath,
    double avgFps);