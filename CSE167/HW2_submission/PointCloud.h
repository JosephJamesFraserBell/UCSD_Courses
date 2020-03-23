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
	GLuint vao, vbo, vbon, ebo;
	GLfloat pointSize;
	
	
public:
	PointCloud(std::string objFilename, GLfloat pointSize);
	
	~PointCloud();
	std::vector<glm::vec3> points;
	std::vector<glm::vec3> normals;
	std::vector<unsigned int> faces;
	void draw();
	void update();
	void updatePointSize(GLfloat size);
	void spin(float deg);
	void incrementX();
	void incrementY();
	void incrementZ();
	void decrementX();
	void decrementY();
	void decrementZ();
	int getXincrement();
	int getYincrement();
	int getZincrement();
	void resetIncrements();
};

#endif

