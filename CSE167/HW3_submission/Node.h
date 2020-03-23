#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
class Node {
public:
	virtual void draw(glm::mat4 C) = 0;
	virtual void update(glm::vec3 translation, int dir) = 0;

};
