#include "BezierCurve.h"

void BezierCurve::draw() {

	/*
	glBindVertexArray(vao);
	glLineWidth(2.0f);
	glDrawArrays(GL_LINE_STRIP, 0, points.size());
	glBindVertexArray(0);
	/**/
	glBegin(GL_LINES);
	for (int i = 0; i < points.size(); i++) {
		glVertex3f(points[i][0], points[i][1], points[i][2]);
	}
	glEnd();
	
}

BezierCurve::BezierCurve(glm::mat4 control_points) {
	
	
	
	glm::vec4 b0 = glm::vec4(-1.0f, 3.0f, -3.0f, 1.0f);
	glm::vec4 b1 = glm::vec4(3.0f, -6.0f, 3.0f, 0.0f);
	glm::vec4 b2 = glm::vec4(-3.0f, 3.0f, 0.0f, 0.0f);
	glm::vec4 b3 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
	B_Bez[0] = b0;
	B_Bez[1] = b1;
	B_Bez[2] = b2;
	B_Bez[3] = b3;
	
	for (float i = 0.0f; i < 1.0-1.0f/150.0f; i = i + 1.0f/150.0f) {
		glm::vec4 pt = control_points * B_Bez * glm::vec4(pow(i, 3), pow(i, 2), i, 1);
		points.push_back(glm::vec3(pt));
	}

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * points.size(), points.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}
