#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// set 0
layout(set = 0, binding = 0) uniform sampler2D uSrc0;   // glTF lighting/composite (LINEAR HDR)
layout(set = 0, binding = 1) uniform sampler2D uSrc1;   // 3DGS (LINEAR, premultiplied in rgb, coverage in a)
layout(set = 0, binding = 2) uniform sampler2D uDepth;  // glTF depth (unused in single mode)
layout(set = 0, binding = 3) uniform sampler2D uDepthB; // 3DGS depth01 (unused in single mode)

// mode: 0=gltf only, 1=3dgs only
layout(push_constant) uniform PC
{
    int   mode;   // 0/1/2
    int   mixOp;         // 0/1
    float mixFactor;     // 0..1
    float alphaPow;      // >=1

    float featherRange;  // depth feather range
    float depthEps;      // depth epsilon
    int   useMinDepthA;  // 0/1
    int   _pad0;
} pc;

// ---- tonemap (ACES) ----
vec3 ACESFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

vec3 toDisplay(vec3 linearHdr)
{
    vec3 m = ACESFilm(max(linearHdr, vec3(0.0)));
    return pow(m, vec3(1.0/2.2));
}

void main()
{
    // ==========
    // glTF only
    // ==========
    if (pc.mode == 0)
    {
        vec3 A = texture(uSrc0, vUV).rgb;   // LINEAR HDR
        outColor = vec4(A, 1.0);
        return;
    }

    // ==========
    // 3DGS only
    // ==========
    if (pc.mode == 1)
    {
        vec4 B = texture(uSrc1, vUV);
		outColor = vec4(B.rgb, 1.0);
        return;
    }

    vec3 A = texture(uSrc0, vUV).rgb;
    outColor = vec4(A, 1.0);
}
