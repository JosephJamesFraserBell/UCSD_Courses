#version 330 core
out vec4 fragColor;

uniform sampler2D tex;
uniform int sandFlag;

in vec2 TexCoords;
in vec3 normalOut;

void main()
{
    if (sandFlag == 1) {
        fragColor = texture(tex, TexCoords);
    } else {
        fragColor = vec4(normalOut, 1.0f);
    }
}
