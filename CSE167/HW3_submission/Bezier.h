#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>
#include "glm/gtx/string_cast.hpp"
class Bezier{
public:
	void addChild(Bezier* child);
	virtual void draw();
	Bezier();

private:
	std::vector<Bezier*>::iterator citer;
	std::vector<Bezier*> children;
};