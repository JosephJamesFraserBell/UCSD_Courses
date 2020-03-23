#include "PointCloud.h"

PointCloud::PointCloud(std::string objFilename)
{
	FILE* fp;

	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float xn = 0.0f;
	float yn = 0.0f;
	float zn = 0.0f;
	float xt = 0.0f;
	float yt = 0.0f;
	unsigned int v1 = 0;
	unsigned int v2 = 0;
	unsigned int v3 = 0;
	unsigned int temp = 0;

	char c1 = 'a';
	char c2 = 'a';
	char delim = '/';

	fp = fopen(objFilename.c_str(), "r");

	while (c1 != EOF) {
		c1 = fgetc(fp);
		if (c1 == 'v') {
			c2 = fgetc(fp);
			if (c2 == ' ') {
				fscanf(fp, "%f %f %f", &x, &y, &z);
				points.push_back(glm::vec3(x, y, z));
			}
			else if (c2 == 'n') {
				fscanf(fp, "%f %f %f", &xn, &yn, &zn);
				normals.push_back(glm::vec3(xn, yn, zn));
			}
			else if (c2 == 't') {
				fscanf(fp, "%f %f", &xt, &yt);
				texPoints.push_back(glm::vec2(xt, yt));
			}
		}
		else if (c1 == 'f') {
			c2 = fgetc(fp);
			if (c2 == ' ') {
				fscanf(fp, "%d%c%d%c%d %d%c%d%c%d %d%c%d%c%d", &v1, &delim, &temp, &delim, &temp, &v2, &delim, &temp, &delim, &temp, &v3, &delim, &temp, &delim, &temp);
				indices.push_back((int)v1 - 1);
				indices.push_back((int)v2 - 1);
				indices.push_back((int)v3 - 1);
			}
		}	
	}

	fclose(fp);

	std::cout << "Points: " << points.size() << std::endl;
	std::cout << "Indices: " << indices.size() << std::endl;
	std::cout << "Normals: " << normals.size() << std::endl;
	std::cout << "Tex: " << texPoints.size() << std::endl;
	model = glm::mat4(1);
	color = glm::vec3(1, 0, 0);

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &vbon);
	glGenBuffers(1, &ebo);
	glGenBuffers(1, &vbot);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * points.size(), points.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbon);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normals.size(), normals.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbot);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * texPoints.size(), texPoints.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), indices.data(), GL_STATIC_DRAW);

	// Unbind from the VBO.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Unbind from the VAO.
	glBindVertexArray(0);
	
}

PointCloud::~PointCloud()
{
	// Delete the VBO and the VAO.
	// Failure to delete your VAOs, VBOs and other data given to OpenGL
	// is dangerous and may slow your program and cause memory leaks
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
}

void PointCloud::draw()
{
	// Bind VAO
	glBindVertexArray(vao);

	//glDrawArrays(GL_POINTS, 0, points.size());
	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
	// Unbind VAO
	glBindVertexArray(0);
}

void PointCloud::update()
{
	// Spin the cube by 1 degree.
	spin(0.1f);
	/*
	 * TODO: Section 3: Modify this function to spin the dragon and bunny about
	 * different axes. Look at the spin function for an idea
	 */
}



void PointCloud::spin(float deg)
{
	// Update the model matrix by multiplying a rotation matrix
	model = glm::rotate(model, glm::radians(deg), glm::vec3(0.0f, 1.0f, 0.0f));
}

