#pragma once
#include "Bezier.h"

class BezierCurve : public Bezier {
public:
	void draw();
	GLuint vao, vbo;
	std::vector<glm::vec3> points;
	BezierCurve(glm::mat4 control_points);
	glm::mat4 B_Bez = glm::mat4(0.0f);
	
};