#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>

//out vec2 TexCoords;
out vec4 ParticleColor;

uniform mat4 projection;
uniform mat4 model;
uniform mat4 view;
uniform vec3 offset;
uniform vec4 color;
uniform int flag;

void main()
{
    float scale = 0.8f;
    //TexCoords = vertex.zw;
    ParticleColor = color;
    if(flag == 1)
    {
        gl_Position = projection * view * model * (vec4(vertex.xy * scale, 0.0, 0.0)+vec4(offset, 1.0f));
    }
    else
    {
        gl_Position = projection * view * model * (vec4(0.0f, vertex.y*scale, vertex.x * scale, 0.0)+vec4(offset, 1.0f));
    }
   
 
}
