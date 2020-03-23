#ifndef _POINT_CLOUD_H_
#define _POINT_CLOUD_H_

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <string>

#include "Object.h"

class PointCloud : public Object
{
private:
	std::vector<glm::vec3> points;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texPoints;
	std::vector<unsigned int> indices;
	GLuint vao, vbo, vbon, ebo, vbot;
public:
	PointCloud(std::string objFilename);
	~PointCloud();

	void draw();
	void update();
	void spin(float deg);
};

#endif

