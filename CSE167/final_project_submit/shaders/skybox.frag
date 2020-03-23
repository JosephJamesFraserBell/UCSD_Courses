#version 330 core
out vec4 fragColor;

uniform samplerCube skybox;

in vec3 TexCoords;

void main()
{
    // Use the color passed in. An alpha of 1.0f means it is not transparent.
    fragColor = texture(skybox, TexCoords);
}
