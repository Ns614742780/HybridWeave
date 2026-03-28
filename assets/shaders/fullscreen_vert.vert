#version 450

layout(location = 0) out vec2 vUV;

void main()
{
    // Fullscreen triangle (no vertex buffer needed)
    vec2 pos[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);

    // Map NDC to UV
    vUV = gl_Position.xy * 0.5 + 0.5;
}
