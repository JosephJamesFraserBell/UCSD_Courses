#version 330 core
// NOTE: Do NOT use any version older than 330! Bad things will happen!

// This is an example vertex shader. GLSL is very similar to C.
// You can define extra functions if needed, and the main() function is
// called when the vertex shader gets run.
// The vertex shader gets called once per vertex.

layout (location = 0) in vec3 position;

out vec3 TexCoords;

uniform mat4 projectionSky;
uniform mat4 viewSky;

void main()
{
    TexCoords = position;
    
    vec4 pos = projectionSky * viewSky * vec4(position, 1.0);
    gl_Position = pos.xyww;
    
}
