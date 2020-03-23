#version 330 core
// This is a sample fragment shader.

// Inputs to the fragment shader are the outputs of the same name from the vertex shader.
// Note that you do not have access to the vertex shader's default output, gl_Position.

uniform vec3 color;
uniform vec3 cameraPos;


uniform sampler2D ourTexture;


in vec3 FragPos;
in vec2 texCoord;
out vec4 fragColor;

vec3 norm2 = vec3(0.0, 1.0, 0.0);
uniform samplerCube skybox;

void main()
{
    
    vec3 I = normalize(cameraPos - FragPos);
    vec3 R = -reflect(I, normalize(norm2));
    fragColor = vec4(0.6*texture(ourTexture, texCoord)) + vec4(0.4*texture(skybox, R).rgb, 1.0f);
    
}
