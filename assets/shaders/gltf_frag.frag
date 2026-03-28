#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vWorldNrm;
layout(location = 2) in vec3 vWorldPos;

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

layout(set = 0, binding = 1) uniform sampler2D uTex[];

layout(location = 0) out vec4 outAlbedoRough;   // rgb=albedo(linear), a=roughness
layout(location = 1) out vec4 outNormalMetal;   // rgb=normal(0..1), a=metallic
layout(location = 2) out vec4 outWorldPos;      // xyz=worldPos, w=1
layout(location = 3) out vec4 outEmissiveAO;    // rgb=emissive, a=AO
layout(location = 4) out vec4 outSheen;         // rgb=sheenColor(linear), a=sheenRoughness
layout(location = 5) out uint outMaterial;      // R32_UINT flags
layout(location = 6) out uint outDrawId;		// UINT for software vrs

const uint INVALID_TEX = 0xFFFFFFFFu;

// flags bits (keep consistent with your C++)
const uint MATERIAL_FLAG_SHEEN = 1u << 2;
const uint MATERIAL_FLAG_ALPHA_MASK = 1u << 6;

void main()
{
    // -------- Base Color --------
    vec4 baseColor = pc.baseColorFactor;
    if (pc.baseColorTex != INVALID_TEX) {
        baseColor *= texture(uTex[nonuniformEXT(pc.baseColorTex)], vUV);
    }

    // 可选：alpha mask 提前 discard（如果你已经用 flag 标记）
    // if ((pc.materialFlags & MATERIAL_FLAG_ALPHA_MASK) != 0u) {
    //     if (baseColor.a < 0.5) discard;
    // }

    // -------- Metallic / Roughness --------
    float metallic  = pc.metallicFactor;
    float roughness = pc.roughnessFactor;

    if (pc.mrTex != INVALID_TEX) {
        vec4 mrSample = texture(uTex[nonuniformEXT(pc.mrTex)], vUV);
        roughness *= mrSample.g; // glTF: G=roughness
        metallic  *= mrSample.b; // glTF: B=metallic
    }

    roughness = clamp(roughness, 0.04, 1.0);
    metallic  = clamp(metallic,  0.0,  1.0);

    // -------- Normal --------
    vec3 n = normalize(vWorldNrm);

    // -------- Emissive/AO（先默认；后面你可以接 emissiveTex/occlusionTex）--------
    vec3 emissive = vec3(0.0);
    float ao = 1.0;

    // -------- Sheen (KHR_materials_sheen) --------
    vec3  sheenColor = vec3(0.0);
    float sheenRough = 0.0;

    if ((pc.materialFlags & MATERIAL_FLAG_SHEEN) != 0u)
    {
        sheenColor = pc.sheenColorRoughFactor.xyz;
        sheenRough = pc.sheenColorRoughFactor.w;

        if (pc.sheenColorTex != INVALID_TEX) {
            // sheenColorTexture: sRGB（如果你的纹理 view 用 SRGB 格式，这里采样结果已经是线性）
            sheenColor *= texture(uTex[nonuniformEXT(pc.sheenColorTex)], vUV).rgb;
        }

        if (pc.sheenRoughTex != INVALID_TEX) {
            // sheenRoughnessTexture: linear，通常取 R
            sheenRough *= texture(uTex[nonuniformEXT(pc.sheenRoughTex)], vUV).r;
        }

        sheenColor = clamp(sheenColor, vec3(0.0), vec3(1.0));
        sheenRough = clamp(sheenRough, 0.04, 1.0);
    }

    // -------- Outputs --------
    outAlbedoRough = vec4(baseColor.rgb, roughness);
    outNormalMetal = vec4(n * 0.5 + 0.5, metallic);
    outWorldPos    = vec4(vWorldPos, 1.0);
    outEmissiveAO  = vec4(emissive, ao);
    outSheen       = vec4(sheenColor, sheenRough);
    outMaterial    = pc.materialFlags;
	outDrawId 	   = pc.drawId;
}
