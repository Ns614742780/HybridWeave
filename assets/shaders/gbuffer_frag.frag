#version 450
layout(location=0) in vec2 vUv;
layout(set=1,binding=0) uniform sampler2D baseColorTex;
layout(location=0) out vec4 outColor;

void main(){
    outColor = texture(baseColorTex, vUv);
}
