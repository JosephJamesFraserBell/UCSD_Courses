#ifndef _OBJECT_H_
#define _OBJECT_H_

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <string>
#include <iostream>

class Object
{
protected:
	glm::mat4 model;
	glm::vec3 color;
public:
	glm::mat4 getModel() { return model; }
	std::string objectName;
	float spinCounter = 0;
	void setModel(glm::mat4 inModel) { model = inModel; }
	glm::vec3 getColor() { return color; }
	int x_move = 0;
	int y_move = 0;
	int z_move = 0;
	int n = 1;
	int scaleFactor = 0;
	virtual void draw() = 0;
	virtual void update() = 0;
};

#endif

