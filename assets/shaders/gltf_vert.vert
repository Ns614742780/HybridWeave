#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNrm;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTan;

layout(set = 0, binding = 0, std140) uniform UBO
{
    vec4 camera_position;
    mat4 view_mat;
    mat4 proj_mat;
    mat4 view_proj;
    uint width;
    uint height;
    float tan_fovx;
    float tan_fovy;
    vec4 gaze_params;
} ubo;

layout(push_constant) uniform PC
{
    mat4 model;

    vec4 baseColorFactor;

    float metallicFactor;
    float roughnessFactor;

    uint  baseColorTex;
    uint  mrTex;

    uint  materialFlags;
    uint  drawId;
	uint  _pad2;
	uint  _pad3;

    // ★ sheen
    vec4  sheenColorRoughFactor; // xyz=sheenColorFactor, w=sheenRoughnessFactor
    uint  sheenColorTex;         // sRGB
    uint  sheenRoughTex;         // linear (R)
	uint  _pad4;
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vWorldNrm;
layout(location = 2) out vec3 vWorldPos;

void main()
{
    vec4 worldPos = pc.model * vec4(inPos, 1.0);
    vWorldPos = worldPos.xyz;

    mat3 normalMat = transpose(inverse(mat3(pc.model)));
    vWorldNrm = normalize(normalMat * inNrm);

    vUV = inUV;

    gl_Position = ubo.proj_mat * ubo.view_mat * worldPos;
}
