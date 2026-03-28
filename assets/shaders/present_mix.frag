#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// set0: A/B + depthA + depthB
layout(set = 0, binding = 0) uniform sampler2D uColorA; // glTF (lighting/composite)
layout(set = 0, binding = 1) uniform sampler2D uColorB; // 3DGS (rgba16f, premult in rgb, coverage in a)
layout(set = 0, binding = 2) uniform sampler2D uDepthA; // glTF depth (0..1), far=1
layout(set = 0, binding = 3) uniform sampler2D uDepthB; // 3DGS depth01 (0..1), far=1  

layout(push_constant) uniform PC
{
    int   presentMode;   // 0/1/2
    int   mixOp;         // 0/1
    float mixFactor;     // 0..1
    float alphaPow;      // >=1

    float featherRange;  // depth feather range
    float depthEps;      // depth epsilon
    int   useMinDepthA;  // 0/1
    int   _pad0;
} pc;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);

    vec4 a = texelFetch(uColorA, p, 0); // 认为是线性空间
    vec4 b = texelFetch(uColorB, p, 0); // b.rgb 已经 premult，b.a=coverage(0..1)

    float dA = texelFetch(uDepthA, p, 0).r;
    float dB = texelFetch(uDepthB, p, 0).r;

    const float farDepth = 1.0;       // 你 clear depth = 1.0
    const float eps = 1e-6;

    bool hasA = (abs(dA - farDepth) > 1e-5);
    bool hasB = (b.a > 0.0);

    // 只有一个存在
    if (!hasA && !hasB) { outColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
    if ( hasA && !hasB) { outColor = vec4(a.rgb, 1.0); return; }
    if (!hasA &&  hasB) { outColor = vec4(b.rgb, 1.0); return; }

    // 两者都存在：depth 小者更近（normal Z）
    bool bInFront = (dB + eps < dA);

    if (bInFront) {
        // 3DGS 在前：按 coverage 做 over
        vec3 rgb = b.rgb + (1.0 - b.a) * a.rgb;
        outColor = vec4(rgb, 1.0);
    } else {
        // glTF 在前（默认不透明）：直接用 A
        outColor = vec4(a.rgb, 1.0);
    }
}
