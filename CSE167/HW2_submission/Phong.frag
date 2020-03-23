#version 330 core
// This is a sample fragment shader.

// Inputs to the fragment shader are the outputs of the same name from the vertex shader.
// Note that you do not have access to the vertex shader's default output, gl_Position.

uniform vec3 color;
uniform vec3 lightDir;  
uniform vec3 lightColor;
uniform mat4 model;
uniform vec3 plLocation;
uniform int plFlag;
uniform int dlFlag;
// Material properties
uniform float ambientStrength;
uniform float diffuseStrength;
uniform float specularStrength;

// You can output many things. The first vec4 type output determines the color of the fragment
out vec4 fragColor;

in vec3 FragPos;
in vec3 normalA;


vec3 ambient_color;
vec3 norm;
vec3 lightDirection;
float diff;
vec3 diffuse_color;
vec3 diffuse_color2;
vec3 result;
vec3 viewDir;
vec3 reflectDir;
vec3 specular_color;
vec3 specular_color2;
float k = 2.0f;
float attenuation = 1.0f;
float spec;
void main()
{
    // Use the color passed in. An alpha of 1.0f means it is not transparent.
    
    ambient_color = ambientStrength * lightColor;
    if (dlFlag == 1){
        // if direction light
        norm = normalize(normalA);
        lightDirection = normalize(lightDir);
        diff = max(dot(norm, lightDirection), 0.0);
        diffuse_color = diffuseStrength * diff * lightColor;

        viewDir = normalize(vec3(0.0f, 0.0f, 0.0f) - FragPos);
        reflectDir = reflect(-lightDir, norm); 
        spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        specular_color = specularStrength * spec * lightColor;  
    }
    else{
        diffuse_color = vec3(0.0f);
        specular_color = vec3(0.0f);
    }
    
    if (plFlag == 1){
        // if point light
        norm = normalize(normalA);
        lightDirection = normalize(plLocation - FragPos);
        diff = max(dot(norm, lightDirection), 0.0);
        diffuse_color2 = diffuseStrength * diff * lightColor;

        viewDir = normalize(vec3(0.0f, 0.0f, 20.0f) - FragPos);
        reflectDir = reflect(-lightDirection, norm); 
        spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        specular_color2 = specularStrength * spec * lightColor;   
        attenuation = length(plLocation - FragPos);
    }
    else{
        diffuse_color2 = vec3(0.0f);
        specular_color2 = vec3(0.0f);
    }

    result = ambient_color + diffuse_color + specular_color + (diffuse_color2 + specular_color2)/(k*attenuation);

    fragColor = vec4(result, 1.0f);
}