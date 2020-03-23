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
	float points[108] =
	{
	-1000.0f,  1000.0f, -1000.0f,
	-1000.0f, -1000.0f, -1000.0f,
	 1000.0f, -1000.0f, -1000.0f,
	 1000.0f, -1000.0f, -1000.0f,
	 1000.0f,  1000.0f, -1000.0f,
	-1000.0f,  1000.0f, -1000.0f,

	-1000.0f, -1000.0f,  1000.0f,
	-1000.0f, -1000.0f, -1000.0f,
	-1000.0f,  1000.0f, -1000.0f,
	-1000.0f,  1000.0f, -1000.0f,
	-1000.0f,  1000.0f,  1000.0f,
	-1000.0f, -1000.0f,  1000.0f,

	 1000.0f, -1000.0f, -1000.0f,
	 1000.0f, -1000.0f,  1000.0f,
	 1000.0f,  1000.0f,  1000.0f,
	 1000.0f,  1000.0f,  1000.0f,
	 1000.0f,  1000.0f, -1000.0f,
	 1000.0f, -1000.0f, -1000.0f,

	-1000.0f, -1000.0f,  1000.0f,
	-1000.0f,  1000.0f,  1000.0f,
	 1000.0f,  1000.0f,  1000.0f,
	 1000.0f,  1000.0f,  1000.0f,
	 1000.0f, -1000.0f,  1000.0f,
	-1000.0f, -1000.0f,  1000.0f,

	-1000.0f,  1000.0f, -1000.0f,
	 1000.0f,  1000.0f, -1000.0f,
	 1000.0f,  1000.0f,  1000.0f,
	 1000.0f,  1000.0f,  1000.0f,
	-1000.0f,  1000.0f,  1000.0f,
	-1000.0f,  1000.0f, -1000.0f,

	-1000.0f, -1000.0f, -1000.0f,
	-1000.0f, -1000.0f,  1000.0f,
	 1000.0f, -1000.0f, -1000.0f,
	 1000.0f, -1000.0f, -1000.0f,
	-1000.0f, -1000.0f,  1000.0f,
	 1000.0f, -1000.0f,  1000.0f
	};
	std::vector<std::string> faces
	{
		"right.jpg",
		"left.jpg",
		"top.jpg",
		"base.jpg",
		"front.jpg",
		"back.jpg"
	};
	unsigned int textureID;
	GLuint vao, vbo;
	GLfloat pointSize;
	unsigned int cubemapTexture;
public:
	PointCloud(std::string objFilename, GLfloat pointSize);
	~PointCloud();

	void draw();
	void update();

	void updatePointSize(GLfloat size);
	void spin(float deg);
	unsigned int loadCubemap(std::vector<std::string> faces);
};

#endif

