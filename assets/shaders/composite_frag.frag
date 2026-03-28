#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// Set 0: subsample history (2DArray)
layout(set = 0, binding = 0) uniform sampler2DArray uHistColor0;
layout(set = 0, binding = 1) uniform sampler2DArray uHistColor1;

// Set 1: debug/current inputs
layout(set = 1, binding = 0) uniform usampler2D uValidSparse;
layout(set = 1, binding = 1) uniform usampler2D uReconValid;
layout(set = 1, binding = 2) uniform sampler2D  uAnchorColor;
layout(set = 1, binding = 3) uniform sampler2D  uLightingResolved;
layout(set = 1, binding = 4) uniform sampler2D  uDepth;
layout(set = 1, binding = 5) uniform sampler2D  uNormal;
layout(set = 1, binding = 6) uniform usampler2D uDrawId;

layout(push_constant) uniform PC
{
    int   debugView;
    uint  tileSize;     // 1/2/4
    uint  pingMask;     // bit per phase
    float historyMul;   // 0..1
} pc;

vec3 showMask(uint v) { return (v != 0u) ? vec3(1.0) : vec3(0.0); }

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

vec4 loadHistory(ivec2 p)
{
    uint ts = max(pc.tileSize, 1u);

    uint ax = (ts == 1u) ? 0u : uint(p.x) % ts;
    uint ay = (ts == 1u) ? 0u : uint(p.y) % ts;
    uint phase = ax + ay * ts;

    ivec3 histSz3 = textureSize(uHistColor0, 0); // (w,h,layers)
    ivec2 h = ivec2(p.x / int(ts), p.y / int(ts));
    if (h.x < 0 || h.y < 0 || h.x >= histSz3.x || h.y >= histSz3.y)
        return vec4(0,0,0,0);

    uint prevPing = (pc.pingMask >> phase) & 1u;
    return (prevPing == 0u)
        ? texelFetch(uHistColor0, ivec3(h, int(phase)), 0)
        : texelFetch(uHistColor1, ivec3(h, int(phase)), 0);
}

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);

    vec4 h = loadHistory(p);

    // 0: final
    if (pc.debugView == 0)
    {
        vec3 curr = texelFetch(uLightingResolved, p, 0).rgb;
        uint rv   = texelFetch(uReconValid, p, 0).r;

        float conf = clamp(h.a, 0.0, 1.0);
        float hm   = clamp(pc.historyMul, 0.0, 1.0);
        float wHist = conf * hm; // 0..1

        // 用 historyMul 判断 moving（不引入任何像素级高频 gating）
        bool moving = (hm < 0.75);

        // ---- 关键：rv==0 静止时仍保持你原来的“只用 history”，保证静止稳定/不摩尔纹 ----
        if (rv == 0u)
        {
            if (!moving)
            {
                outColor = vec4(toDisplay(h.rgb), 1.0);
                return;
            }

            float w = min(wHist, 0.15);   // 可扫：0.10~0.25（越小越不拖影）
            vec3 outLin = (w < 0.02) ? curr : mix(curr, h.rgb, w);
            outColor = vec4(toDisplay(outLin), 1.0);
            return;
        }

        // rv==1：只做“统一的、低频的”运动抑制（不会产生摩尔纹）
        if (moving)
        {
            // 1) 限制 history 最大占比，降低滞后/拖影
            wHist = min(wHist, 0.25);

            // 2) 提高 bootstrap 阈值：moving 时更偏向 current
            const float movingBootstrap = 0.15;

            vec3 outLin = (wHist < movingBootstrap) ? curr : mix(curr, h.rgb, wHist);
            outColor = vec4(toDisplay(outLin), 1.0);
            return;
        }
        else
        {
            // ---- 静止：完全保留你的原逻辑（不闪、不摩尔纹）----
            vec3 outLin;
            if (wHist < 0.02) outLin = curr;
            else              outLin = mix(curr, h.rgb, wHist);

            outColor = vec4(toDisplay(outLin), 1.0);
            return;
        }
    }

    if (pc.debugView == 401) { float c = clamp(h.a, 0.0, 1.0); outColor = vec4(c,c,c,1); return; }
    if (pc.debugView == 410) { uint v = texelFetch(uValidSparse, p, 0).r; outColor = vec4(showMask(v),1); return; }
    if (pc.debugView == 411) { uint v = texelFetch(uReconValid,  p, 0).r; outColor = vec4(showMask(v),1); return; }
    if (pc.debugView == 420) { vec3 c = texelFetch(uAnchorColor, p, 0).rgb; outColor = vec4(toDisplay(c),1); return; }
    if (pc.debugView == 430) { vec3 c = texelFetch(uLightingResolved, p, 0).rgb; outColor = vec4(toDisplay(c),1); return; }

    if (pc.debugView == 500) { float d = texelFetch(uDepth, p, 0).r; outColor = vec4(vec3(d),1); return; }
    if (pc.debugView == 501) { vec3 n = texelFetch(uNormal, p, 0).xyz * 0.5 + 0.5; outColor = vec4(n,1); return; }
    if (pc.debugView == 502) { uint id = texelFetch(uDrawId, p, 0).r; float f = fract(float(id)*0.12345); outColor = vec4(f,1.0-f,0.5,1); return; }

    outColor = vec4(1,0,1,1);
}
