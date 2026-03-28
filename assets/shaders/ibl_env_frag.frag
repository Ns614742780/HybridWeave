#version 450

layout(location = 0) in vec2 vUV;

layout(set = 0, binding = 0) uniform sampler2D uEnvEquirect;

layout(push_constant) uniform PC
{
    float exposure;
    float _pad0;
    float _pad1;
    float _pad2;
} pc;

layout(location = 0) out vec4 outColor;

// 简单 Reinhard + gamma
vec3 tonemapReinhard(vec3 x)
{
    return x / (x + vec3(1.0));
}

void main()
{
    vec2 uv = vUV;
    uv.x = fract(uv.x);
    uv.y = clamp(uv.y, 0.0, 1.0);

    vec3 hdr = texture(uEnvEquirect, uv).rgb;

    hdr *= pc.exposure;

    vec3 ldr = tonemapReinhard(hdr);
    // gamma
    ldr = pow(ldr, vec3(1.0 / 2.2));

    outColor = vec4(ldr, 1.0);
}
