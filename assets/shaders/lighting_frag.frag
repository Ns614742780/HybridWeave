#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 vUV;

// ---------------- GBuffer ----------------
layout(set = 0, binding = 0) uniform sampler2D  gAlbedoRough;
layout(set = 0, binding = 1) uniform sampler2D  gNormalMetal;
layout(set = 0, binding = 2) uniform sampler2D  gWorldPos;
layout(set = 0, binding = 3) uniform sampler2D  gEmissiveAO;
layout(set = 0, binding = 4) uniform sampler2D  gSheenColorRough;
layout(set = 0, binding = 5) uniform usampler2D gMaterial;
layout(set = 0, binding = 6) uniform sampler2D  gDepth;

// ---------------- Camera ----------------
layout(set = 2, binding = 0, std140) uniform UBO
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

// ---------------- Light ----------------
layout(set = 1, binding = 0, std140) uniform LightUBO
{
    vec4 lightDir;
    vec4 lightColor;
    vec4 envParams; // x=envIntensity, y=enable, z=prefilterMipLevels
} light;

// ---------------- IBL ----------------
layout(set = 1, binding = 1) uniform samplerCube uEnvCube;
layout(set = 1, binding = 2) uniform samplerCube uIrradianceCube;
layout(set = 1, binding = 3) uniform samplerCube uPrefilterCube;
layout(set = 1, binding = 4) uniform sampler2D   uBrdfLut;

// ★ Sheen IBL resources
layout(set = 1, binding = 5) uniform samplerCube uSheenPrefilterCube;
layout(set = 1, binding = 6) uniform sampler2D   uSheenLut;

layout(push_constant) uniform PC
{
    int debugView;
    int _pad0;
    int _pad1;
    int _pad2;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// ---------------- utils ----------------
float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3 x)  { return clamp(x, vec3(0.0), vec3(1.0)); }

// ---------------- GGX ----------------
float D_GGX(float NdotH, float a)
{
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}
float G_SchlickGGX(float NdotX, float k)
{
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-6);
}
float G_Smith(float NdotV, float NdotL, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return G_SchlickGGX(NdotV, k) * G_SchlickGGX(NdotL, k);
}
vec3 F_Schlick(vec3 F0, float cosTheta)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
vec3 F_SchlickRoughness(vec3 F0, float cosTheta, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
              * pow(1.0 - cosTheta, 5.0);
}

// ---------------- Sheen (Charlie + Neubelt) ----------------
float D_Charlie(float NdotH, float roughness)
{
    float alpha = max(roughness * roughness, 1e-4);
    float invAlpha = 1.0 / alpha;
    float sin2h = max(1.0 - NdotH * NdotH, 1e-4);
    return (2.0 + invAlpha) * pow(sin2h, 0.5 * invAlpha) / (2.0 * PI);
}
float V_Neubelt(float NdotL, float NdotV)
{
    return 1.0 / max(4.0 * (NdotL + NdotV - NdotL * NdotV), 1e-4);
}

// ---------------- tonemap ----------------
vec3 ACESFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

// ========================= main =========================
void main()
{
    vec4 a  = texture(gAlbedoRough, vUV);
    vec4 nm = texture(gNormalMetal, vUV);
    vec4 wp = texture(gWorldPos, vUV);
    vec4 ea = texture(gEmissiveAO, vUV);
    vec4 sh = texture(gSheenColorRough, vUV);

    if (wp.w < 0.5) { outColor = vec4(0.0); return; }

    vec3  albedo    = a.rgb;
    float roughness = clamp(a.w, 0.04, 1.0);
    float metallic  = saturate(nm.w);

    vec3 N = normalize(nm.xyz * 2.0 - 1.0);
    vec3 V = normalize(ubo.camera_position.xyz - wp.xyz);

    ivec2 pix = ivec2(gl_FragCoord.xy);
	uint flags = texelFetch(gMaterial, pix, 0).r;

    bool isCloth        = (flags & (1u<<0)) != 0u;
    bool isFoliage      = (flags & (1u<<1)) != 0u;
    bool hasSheen       = (flags & (1u<<2)) != 0u;
    bool isThin         = (flags & (1u<<3)) != 0u;
    bool hasTransmission= (flags & (1u<<4)) != 0u;
    bool isDoubleSided  = (flags & (1u<<5)) != 0u;

    vec3 N_base = N;
    if (isDoubleSided && dot(N_base, V) < 0.0) N_base = -N_base;

    vec3 N_ibl = N_base;
	vec3 R_ibl = reflect(-V, N_ibl);

    float ao = saturate(ea.a);

    float NdotV_base = saturate(dot(N_base, V));
    float NdotV_ibl  = saturate(dot(N_ibl,  V));

    // -------- sheen params --------
    vec3  sheenColor = vec3(0.0);
    float sheenRough = 0.0;
    float sheenWeight = 0.0;

    if (hasSheen)
    {
        sheenColor  = saturate(sh.rgb);
        sheenRough  = clamp(sh.a, 0.04, 1.0);
        sheenWeight = clamp(max(max(sheenColor.r, sheenColor.g), sheenColor.b), 0.0, 1.0);
    }

    // ========================= Direct =========================
    vec3 Lo = vec3(0.0);

    vec3 L = normalize(-light.lightDir.xyz);
    vec3 H = normalize(V + L);

    float NdotL = saturate(dot(N_base, L));
    float NdotH = saturate(dot(N_base, H));
    float VdotH = saturate(dot(V, H));

    vec3 radiance = light.lightColor.rgb * light.lightColor.w;

    float aGGX = roughness * roughness;
    float D = D_GGX(NdotH, aGGX);
    float G = G_Smith(NdotV_base, NdotL, roughness);
    vec3  F = F_Schlick(mix(vec3(0.04), albedo, metallic), VdotH);

    vec3 specular = (D * G * F) / max(4.0 * NdotV_base * NdotL, 1e-6);

    vec3 kS = F_SchlickRoughness(mix(vec3(0.04), albedo, metallic), NdotV_base, roughness);
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    if (hasSheen) {
        kD *= (1.0 - sheenWeight);
        specular *= (1.0 - 0.65 * sheenWeight);
    }

    if (isFoliage) specular *= 0.35;

    vec3 diffuse = kD * albedo / PI;
    vec3 direct  = (diffuse + specular) * radiance * NdotL;

    if (hasSheen)
    {
        float Dsh = D_Charlie(NdotH, sheenRough);
        float Vsh = V_Neubelt(NdotL, NdotV_base);
        vec3 sheenSpec = sheenColor * Dsh * Vsh * radiance * NdotL;
        if (isCloth) sheenSpec *= 1.15;
        direct += sheenSpec;
    }

    Lo += direct;

    // ========================= IBL =========================
    if (light.envParams.y > 0.5)
    {
        float maxLod = max(light.envParams.z - 1.0, 0.0);
		vec3 F0 = mix(vec3(0.04), albedo, metallic);
		vec3 kS_ibl = F_SchlickRoughness(F0, NdotV_ibl, roughness);
		vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);
		if (hasSheen) kD_ibl *= (1.0 - sheenWeight);
        vec3 irradiance = texture(uIrradianceCube, N_ibl).rgb;
        vec3 diffuseIBL = irradiance * albedo * kD_ibl * ao;

        float lodBase = clamp((roughness * roughness) * maxLod, 0.0, maxLod);
        vec3 prefiltered = textureLod(uPrefilterCube, R_ibl, lodBase).rgb;
        vec2 brdf = texture(uBrdfLut, vec2(NdotV_ibl, roughness)).rg;

        vec3 specIBL = prefiltered * (kS_ibl * brdf.x + brdf.y);
        if (isFoliage) specIBL *= 0.35;
        if (isCloth)   specIBL *= 0.45;

        vec3 sheenIBL = vec3(0.0);
        if (hasSheen)
        {
            float lodSheen = clamp((sheenRough * sheenRough) * maxLod, 0.0, maxLod);
            vec3 preSheen = textureLod(uSheenPrefilterCube, R_ibl, lodSheen).rgb;
            float sheenDFG = texture(uSheenLut, vec2(NdotV_ibl, sheenRough)).r;
            sheenIBL = preSheen * sheenColor * sheenDFG * ao;
            if (isCloth) sheenIBL *= 1.05;
        }

        Lo += (diffuseIBL + specIBL + sheenIBL) * light.envParams.x;
    }

    Lo += ea.rgb;

    vec3 color = ACESFilm(Lo);
    color = pow(max(color, 0.0), vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
