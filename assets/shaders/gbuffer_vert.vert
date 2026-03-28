#version 450
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNrm;
layout(location=2) in vec2 inUv;

layout(set=0,binding=0) uniform UBO {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
} ubo;

layout(location=0) out vec2 vUv;

void main(){
    vUv = inUv;
    gl_Position = ubo.viewProj * vec4(inPos,1.0);
}
