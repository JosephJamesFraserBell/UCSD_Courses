#include "DirectionalLight.h"


DirectionalLight::DirectionalLight(glm::vec3 inColor, glm::vec3 inDirection) {
	color = inColor;
	direction = inDirection;
}