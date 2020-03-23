#version 330 core
// NOTE: Do NOT use any version older than 330! Bad things will happen!

// This is an example vertex shader. GLSL is very similar to C.
// You can define extra functions if needed, and the main() function is
// called when the vertex shader gets run.
// The vertex shader gets called once per vertex.

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoords;
layout (location = 2) in vec3 normal;

uniform mat4 projectionGround;
uniform mat4 viewGround;
uniform mat4 modelGround;

out vec2 TexCoords;
out vec3 normalOut;

void main()
{
    gl_Position = projectionGround * viewGround * modelGround * vec4(position, 1.0f);
    
    TexCoords = texCoords;
    normalOut = (0.5f * normalize(normal)) + 0.5f;
}
