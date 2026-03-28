#include "Camera.h"
#include <fstream>
#include <iomanip>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <filesystem>

void appendCameraLog(
    const CameraSnapshot& snap,
    const std::string& filepath,
    double avgFps)
{
    std::ofstream ofs(filepath, std::ios::out | std::ios::app);
    if (!ofs.is_open())
        throw std::runtime_error("Failed to open log file: " + filepath);

    ofs << std::fixed << std::setprecision(6);

    ofs << "---- Camera Snapshot ----\n";
    ofs << "avg_fps " << avgFps << "\n";

    ofs << "position ("
        << snap.position.x << "f, "
        << snap.position.y << "f, "
        << snap.position.z << "f)\n";

    ofs << "rotation ("
        << snap.rotation.w << "f, "
        << snap.rotation.x << "f, "
        << snap.rotation.y << "f, "
        << snap.rotation.z << "f)\n";

    ofs << "fov_y_deg  " << snap.fov_y_deg << "\n";
    ofs << "near_plane " << snap.near_plane << "\n";
    ofs << "far_plane  " << snap.far_plane << "\n";
    ofs << "aspect     " << snap.aspect << "\n\n";

    ofs.flush();
}

