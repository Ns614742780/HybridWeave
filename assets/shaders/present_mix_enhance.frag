#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D uColorA;   // glTF (A)
layout(set = 0, binding = 1) uniform sampler2D uColorB;   // 3DGS (B) premult rgb, a=coverage
layout(set = 0, binding = 2) uniform sampler2D uDepthA;
layout(set = 0, binding = 3) uniform sampler2D uDepthB;

layout(std430, set = 1, binding = 0) readonly buffer AutoMatchBuf
{
    vec4 params;     // (gain, wbR, wbG, wbB)
    vec4 blurStats;  // (blurSigma, blurRatio, EA, EB)
} gAuto;

layout(set = 1, binding = 1) uniform sampler2D uLut;

// blurMap (per-pixel stable soften weight)
layout(set = 2, binding = 0) uniform sampler2D uBlurMap;

layout(push_constant) uniform PC
{
    int   presentMode;   // 0/1/2
    int   mixOp;         // debug branch
    float mixFactor;     // fusion strength (0..1)
    float alphaPow;      // >=1

    float featherRange;
    float depthEps;
    int   useMinDepthA;
    int   styleLock;
} pc;

const float FAR_DEPTH = 1.0;

float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3  clamp01(vec3 x)  { return clamp(x, vec3(0.0), vec3(1.0)); }
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

bool depthValid01(float d)
{
    return (d > 1e-6) && (d < 1.0 - 1e-6);
}

bool hasGeomA(float dA)
{
    // A has geometry if depth is valid and not far plane
    return depthValid01(dA) && (abs(dA - FAR_DEPTH) > 1e-5);
}

bool hasAnyB(vec4 b)
{
    // B existence purely by alpha>0
    return (b.a > 0.0);
}

float depthTol_soft(float d)
{
    return 0.0015 + 0.020 * d;
}

float depthTol_hard(float d)
{
    return 0.0010 + 0.010 * d;
}

float hardBOccludesA(float dA, float dB)
{
    if (!depthValid01(dA) || !depthValid01(dB)) return 0.0;
    float tol = depthTol_hard(min(dA, dB));
    return (dB < (dA - tol)) ? 1.0 : 0.0;
}

float frontWeight(float dA, float dB)
{
    if (!depthValid01(dA) || !depthValid01(dB))
        return 0.0;

    float sep = dA - dB;
    float tol = depthTol_soft(max(dA, dB));
    float w = smoothstep(pc.depthEps, pc.depthEps + tol, sep);
    return saturate(w);
}

float minDepth3x3(sampler2D dtex, ivec2 p)
{
    float m = 1.0;
    for (int oy=-1; oy<=1; ++oy)
    for (int ox=-1; ox<=1; ++ox)
        m = min(m, texelFetch(dtex, p + ivec2(ox,oy), 0).r);
    return m;
}

float sigmaToStrength(float sigmaPx)
{
    float t = saturate(sigmaPx / 3.0);
    return t * (2.0 - t);
}

vec3 blur5_raw(sampler2D tex, ivec2 p)
{
    vec3 c0 = texelFetch(tex, p, 0).rgb;
    vec3 c1 = texelFetch(tex, p + ivec2( 1, 0), 0).rgb;
    vec3 c2 = texelFetch(tex, p + ivec2(-1, 0), 0).rgb;
    vec3 c3 = texelFetch(tex, p + ivec2( 0, 1), 0).rgb;
    vec3 c4 = texelFetch(tex, p + ivec2( 0,-1), 0).rgb;
    return (c0 + c1 + c2 + c3 + c4) * 0.2;
}

vec3 unsharpKillA_fullscreen(ivec2 p, float w)
{
    vec3 c0 = texelFetch(uColorA, p, 0).rgb;
    vec3 cB = blur5_raw(uColorA, p);
    vec3 hi = c0 - cB;

    hi = clamp(hi, vec3(-0.22), vec3(0.22));
    vec3 outc = c0 - hi * (1.35 * w);
    return clamp(outc, vec3(0.0), vec3(1.0));
}

vec3 blurA_aggressive13tap(ivec2 p, float radiusPx)
{
    float r = clamp(radiusPx, 0.0, 6.0);
    if (r < 0.5) return texelFetch(uColorA, p, 0).rgb;

    int R = int(floor(r + 0.5));

    vec3 c = vec3(0.0);
    float wsum = 0.0;

    {   float w = 0.22;
        c += texelFetch(uColorA, p, 0).rgb * w;
        wsum += w;
    }

    {   float w = 0.13;
        c += texelFetch(uColorA, p + ivec2( R, 0), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2(-R, 0), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2( 0, R), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2( 0,-R), 0).rgb * w;
        wsum += 4.0 * w;
    }

    {   float w = 0.09;
        c += texelFetch(uColorA, p + ivec2( R, R), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2(-R, R), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2( R,-R), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2(-R,-R), 0).rgb * w;
        wsum += 4.0 * w;
    }

    int R2 = min(6, R * 2);
    if (R2 >= 2)
    {
        float w = 0.04;
        c += texelFetch(uColorA, p + ivec2( R2, 0), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2(-R2, 0), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2( 0, R2), 0).rgb * w;
        c += texelFetch(uColorA, p + ivec2( 0,-R2), 0).rgb * w;
        wsum += 4.0 * w;
    }

    return c * (1.0 / max(wsum, 1e-6));
}

vec3 applyAutoWB_Stable(vec3 rgb)
{
    if (pc.styleLock != 0) return rgb;
    vec3 wb = gAuto.params.yzw;
    if (all(equal(wb, vec3(0.0)))) wb = vec3(1.0);
    return rgb * wb;
}

vec3 applyLutLuma(vec3 c)
{
    c = clamp(c, vec3(0.0), vec3(1.0));
    float Lin = max(luma(c), 1e-5);
    float y = texture(uLut, vec2(Lin, 0.5)).r;
    float Lout = max(y, 1e-5);
    float s = Lout / Lin;
    return clamp(c * s, vec3(0.0), vec3(1.0));
}

float detailMaskFromA(ivec2 p)
{
    vec3 c0 = texelFetch(uColorA, p, 0).rgb;
    vec3 cx = texelFetch(uColorA, p + ivec2(1,0), 0).rgb;
    vec3 cy = texelFetch(uColorA, p + ivec2(0,1), 0).rgb;
    vec3 c1 = texelFetch(uColorA, p + ivec2(-1,0), 0).rgb;
    vec3 c2 = texelFetch(uColorA, p + ivec2(0,-1), 0).rgb;

    float L0 = luma(c0);
    float gx = 0.5 * (luma(cx) - luma(c1));
    float gy = 0.5 * (luma(cy) - luma(c2));
    float g  = sqrt(gx*gx + gy*gy);

    float mu = (L0 + luma(cx) + luma(c1) + luma(cy) + luma(c2)) * 0.2;
    float v0 = (L0 - mu);
    float v1 = (luma(cx) - mu);
    float v2 = (luma(c1) - mu);
    float v3 = (luma(cy) - mu);
    float v4 = (luma(c2) - mu);
    float var = (v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4) * 0.2;

    float sig = sqrt(max(var, 0.0));
    float hf = 0.65 * g + 0.85 * sig;
    float m = smoothstep(0.015, 0.075, hf);

    float dark = 1.0 - smoothstep(0.03, 0.12, L0);
    m *= (1.0 - 0.35 * dark);

    return saturate(m);
}

// strict premult composition when B in front
vec3 mixHardOcclusion(vec4 a, vec4 b, float dA, float dB)
{
    const float eps = 1e-6;
    bool bInFront = (dB + eps < dA);
    if (bInFront) return b.rgb + (1.0 - b.a) * a.rgb;
    return a.rgb;
}

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);

    vec4 a = texelFetch(uColorA, p, 0);
    vec4 b = texelFetch(uColorB, p, 0);

    if (pc.presentMode == 0) { outColor = vec4(a.rgb, 1.0); return; }
    if (pc.presentMode == 1) { outColor = vec4(b.rgb, 1.0); return; }

    float dA = texelFetch(uDepthA, p, 0).r;
    float dB = texelFetch(uDepthB, p, 0).r;

    bool  hasA = hasGeomA(dA);
    bool  hasB = hasAnyB(b);

    // Nothing at all
    if (!hasA && !hasB) { outColor = vec4(0,0,0,1); return; }

    // If only B exists
    if (!hasA && hasB) { outColor = vec4(b.rgb, 1.0); return; }

    // ====== from here: hasA is true => ALWAYS compute A style (decoupled) ======
    float dA_cmp = dA;
    if (pc.useMinDepthA != 0) dA_cmp = minDepth3x3(uDepthA, p);

    float fusion = clamp(pc.mixFactor, 0.0, 1.0);

    // =========
    // STYLE PASS (A only, FULLSCREEN on A pixels, DECOUPLED from B)
    // =========
    float sigma = gAuto.blurStats.x;
    if (!(sigma >= 0.0)) sigma = 0.0;
    sigma = clamp(sigma, 0.0, 3.0);

    float blurRatio = gAuto.blurStats.y;
    if (!(blurRatio >= 0.0)) blurRatio = 0.0;
    blurRatio = clamp(blurRatio, 0.0, 1.0);

    float blurStrength = sigmaToStrength(sigma);

    float blurW = texelFetch(uBlurMap, p, 0).r;
    blurW = clamp(blurW, 0.0, 1.0);
    float blurW_boost = saturate(blurW * 1.60);

    float detailMask = detailMaskFromA(p);

    float globalSoft_sigma = fusion * blurStrength * (0.55 + 0.80 * blurRatio);
    float globalSoft_map   = fusion * blurW_boost;
    float baseline         = fusion * (0.05 + 0.14 * blurW_boost);

    float globalSoft = max(globalSoft_sigma, globalSoft_map);
    globalSoft = max(globalSoft, baseline);
    globalSoft = clamp(globalSoft, 0.0, 1.05);

    float AisFar = smoothstep(0.70, 0.98, dA_cmp);
    globalSoft *= (1.0 - 0.45 * AisFar);

    float blurKillTex = 1.0 - 0.72 * detailMask;
    blurKillTex = clamp(blurKillTex, 0.20, 1.0);
    globalSoft *= blurKillTex;

    if (pc.mixOp == 23) { outColor = vec4(vec3(clamp(globalSoft,0.0,1.0)), 1.0); return; }

    vec3 A_soft_raw = unsharpKillA_fullscreen(p, saturate(globalSoft));

    float radius_sigma = 1.0 + 2.8 * blurStrength * (0.45 + 0.55 * blurRatio);
    float radius_map   = 0.8 + 5.5 * blurW_boost;
    float radiusPx = max(radius_sigma, radius_map);
    radiusPx *= fusion;
    radiusPx = max(radiusPx, 1.35 * fusion);
    radiusPx = clamp(radiusPx, 0.0, 6.0);

    vec3 A_blur_raw = blurA_aggressive13tap(p, radiusPx);

    vec3 rgbA_raw  = mix(A_soft_raw, A_blur_raw, 0.65);
    vec3 rgbA_wb   = applyAutoWB_Stable(rgbA_raw);
    vec3 rgbA_corr = applyLutLuma(rgbA_wb);

    vec3 rgbA_style = mix(rgbA_wb, rgbA_corr, 0.35 + 0.45 * fusion);
	rgbA_style = clamp01(rgbA_style);

	// ============================================================
	// STYLE ENERGY COMPENSATION (fix darkening)
	// ============================================================
	// Compare luminance between original A and stylized output, then compensate.
	// IMPORTANT: only compensates when style strength is on.
	float Lsrc  = max(luma(a.rgb), 1e-4);
	float Lsty  = max(luma(rgbA_style), 1e-4);

	// gain to restore energy
	float gainE = clamp(Lsrc / Lsty, 0.85, 1.45);

	// strength: depends on style amount (globalSoft ~ style coverage)
	float styleW = clamp(globalSoft, 0.0, 1.0);

	// do not boost highly detailed pixels too much (avoid noisy brightening)
	float detW = clamp(detailMask, 0.0, 1.0);
	float protectDetail = mix(1.0, 0.55, detW);  // detailed => less compensation

	// final compensation amount
	float compW = fusion * styleW * protectDetail;

	// apply
	rgbA_style *= mix(1.0, gainE, 0.75 * compW);

	// small toe lift to avoid muddy shadows after blur
	float toe = 0.06 * compW;
	rgbA_style = rgbA_style + toe * (1.0 - rgbA_style);

	rgbA_style = clamp01(rgbA_style);

    // If B doesn't exist at this pixel, we are DONE: output styled glTF.
    if (!hasB) { outColor = vec4(rgbA_style, 1.0); return; }

    // =========
    // OCCLUSION / B resolve (ONLY controls who covers who)
    // =========
    float alpha0 = clamp(b.a, 0.0, 1.0);
    float frontW   = frontWeight(dA_cmp, dB);
    float bOccHard = hardBOccludesA(dA_cmp, dB);

    // geometry correct minimum
    vec3 rgbHard = mixHardOcclusion(a, b, dA_cmp, dB);

    // If B is not in front, just use styled A (遮挡正确)
    bool bFront = (dB + 1e-6 < dA_cmp);
    if (!bFront) { outColor = vec4(rgbA_style, 1.0); return; }

    // If B is in front: enforce non-transparent occlusion feeling.
    // IMPORTANT: do NOT let low alpha make B "see-through" too much when depth says it's front.
    float alphaEff = alpha0;

    // Hard-front => push alpha toward solid
    float hardSolid = saturate(bOccHard);
    alphaEff = mix(alphaEff, 1.0, 0.65 * hardSolid);

    // also ensure minimum solidity when B in front but alpha is small (avoid "transparent splats")
    alphaEff = max(alphaEff, 0.35 * frontW);

    vec3 rgb = b.rgb + (1.0 - alphaEff) * rgbA_style;

    // Blend back to strict geometry composite for stability
    float geoMix = mix(0.45, 1.0, hardSolid);
    rgb = mix(rgb, rgbHard, geoMix * (1.0 - saturate(alpha0)));

    outColor = vec4(clamp01(rgb), 1.0);
}
