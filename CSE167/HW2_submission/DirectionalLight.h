#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <string>

class DirectionalLight {
public:
	glm::vec3 color;
	glm::vec3 direction;

	DirectionalLight(glm::vec3 inColor, glm::vec3 inDirection);
};