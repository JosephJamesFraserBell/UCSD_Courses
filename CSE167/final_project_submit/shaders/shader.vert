#version 330 core
// NOTE: Do NOT use any version older than 330! Bad things will happen!

// This is an example vertex shader. GLSL is very similar to C.
// You can define extra functions if needed, and the main() function is
// called when the vertex shader gets run.
// The vertex shader gets called once per vertex.

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform vec3 color;
uniform float t;
out vec3 FragPos;
out vec2 texCoord;
vec3 pos2;

float frequency = 40.0;
float amplitude = 1.0;
float timestep = 300.0;
vec3 dv_dx;
vec3 dv_dz;
vec3 n;
void main()
{
    
    
    pos2 = position;
    pos2.y = 60.0* sin(10 * position.x + t/timestep ) * sin(5 * position.x + t/timestep )*sin(20* position.x + t/timestep ) *cos(10 * position.z + t/timestep)*cos(5 * position.z + t/timestep);
    
    gl_Position = projection * view * model * vec4(pos2, 1.0);
    texCoord = aTexCoord;
    FragPos = vec3(model * vec4(position, 1.0));
  
}
