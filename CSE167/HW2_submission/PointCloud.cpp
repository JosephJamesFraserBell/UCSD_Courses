#include "PointCloud.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <glm/gtx/string_cast.hpp >
PointCloud::PointCloud(std::string objFilename, GLfloat pointSize)
	: pointSize(pointSize)
{


	/*
	 * TODO: Section 2: Currently, all the points are hard coded below.
	 * Modify this to read points from an obj file.
	 * Don't forget to load in the object normals for normal coloring as well
	 */
	FILE* fp;
	
	objectName = objFilename;
	float maxX = -100.0f;
	float maxY = -100.0f;
	float maxZ = -100.0f;
	float minX = 100.0f;
	float minY = 100.0f;
	float minZ = 100.0f;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float xn = 0.0f;
	float yn = 0.0f;
	float zn = 0.0f;
	unsigned int v1 = 0;
	unsigned int v2 = 0;
	unsigned int v3 = 0;
	char delim = '/';
	//float r, g, b;
	char c1 = 'a';
	char c2 = 'a';


	std::string objName; 

	//Reading in .obj file
	fp = fopen(objectName.c_str(), "r");

	int vertices = 34834;
	int counter = 0;
	
	while (c1!=EOF) {
		c1 = fgetc(fp);
		if (c1 == 'v') {
			c2 = fgetc(fp);
			if (c2 == ' ') {
				fscanf(fp, "%f %f %f", &x, &y, &z);
				points.push_back(glm::vec3(x, y, z));
				if (x > maxX) {
					maxX = x;
				}
				else if (x < minX) {
					minX = x;
				}
				if (y > maxY) {
					maxY = y;
				}
				else if (y < minY) {
					minY = y;
				}
				if (z > maxZ) {
					maxZ = z;
				}
				else if (z < minZ) {
					minZ = z;
				}
			}
			else if (c2 == 'n') {
				fscanf(fp, "%f %f %f", &xn, &yn, &zn);
				normals.push_back(glm::vec3(xn, yn, zn));
			}
		}
		else if (c1 == 'f') {
			c2 = fgetc(fp);
			if (c2 == ' ') {
				fscanf(fp, "%d%c%c%d %d%c%c%d %d%c%c%d", &v1, &delim, &delim, &v1, &v2, &delim, &delim, &v2, &v3, &delim, &delim, &v3);
				faces.push_back((int) v1 - 1);
				faces.push_back((int) v2 - 1);
				faces.push_back((int) v3 - 1);
			}
			
		}
		
	}
	
	float adjustmentX = (maxX + minX) / 2.0f;
	float adjustmentY = (maxY + minY) / 2.0f;
	float adjustmentZ = (maxZ + minZ) / 2.0f;
	/*
	std::cout << maxX << "," << minX << std::endl;
	std::cout << maxY << "," << minY << std::endl;
	std::cout << maxZ << "," << minZ << std::endl;
	std::cout <<adjustmentX << std::endl;
	std::cout << adjustmentY << std::endl;
	std::cout <<adjustmentZ << std::endl;
	/**/
	maxX = maxX - adjustmentX;
	maxY = maxY - adjustmentY;
	maxZ = maxZ - adjustmentZ;

	for (int i = 0; i < points.size(); i++) {
		if (maxX > 1.0f) {
			points[i][0] = (points[i][0] - adjustmentX) / maxX;
		}
		else {
			points[i][0] = (points[i][0] - adjustmentX);
		}
		if (maxY > 1.0f) {
			points[i][1] = (points[i][1] - adjustmentY) / maxY;
		}
		else {
			points[i][1] = (points[i][1] - adjustmentY);
		}
		if (maxY > 1.0f) {
			points[i][2] = (points[i][2] - adjustmentZ) / maxZ;
		}
		else {
			points[i][2] = (points[i][2] - adjustmentZ);
		}
		
		
	}
	/*
	float tempMaxX = -100.0f;
	float tempMinX = 100.0f;
	float tempMaxY = -100.0f;
	float tempMinY = 100.0f;
	float tempMaxZ = -100.0f;
	float tempMinZ = 100.0f;
	for (int i = 0; i < points.size(); i++) {

		float xx = points[i][0];
		float yy = points[i][1];
		float zz = points[i][2];
		if (xx > tempMaxX) {
			tempMaxX = xx;
		}
		else if (xx < tempMinX) {
			tempMinX = xx;
		}
		if (yy > tempMaxY) {
			tempMaxY = yy;
		}
		else if (yy < tempMinY) {
			tempMinY = yy;
		}
		if (zz > tempMaxZ) {
			tempMaxZ = zz;
		}
		else if (zz < tempMinZ) {
			tempMinZ = zz;
		}
	}

	std::cout << tempMaxX << "," << tempMinX << std::endl;
	std::cout << tempMaxY << "," << tempMinY << std::endl;
	std::cout << tempMaxZ << "," << tempMinZ << std::endl;
	/**/
	fclose(fp);

	
	model = glm::mat4(1);
	color = glm::vec3(0.5f, 0.5f, 0.5f);

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &vbon);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * points.size(), points.data(), GL_STATIC_DRAW);
	
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);


	glBindBuffer(GL_ARRAY_BUFFER, vbon);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normals.size(), normals.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * faces.size(), faces.data(), GL_STATIC_DRAW);
	// Unbind everything
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	/*
	for (int i = 0; i < faces.size(); i++) {
		std::cout << faces[i] << std::endl;
	}
	/**/

}

void PointCloud::incrementX() {

	x_move++;
}

void PointCloud::incrementY() {

	y_move++;
}

void PointCloud::incrementZ() {
	z_move++;
}

void PointCloud::decrementX() {

	x_move--;
}

void PointCloud::decrementY() {
	y_move--;
}

void PointCloud::decrementZ() {

	z_move--;
}

int PointCloud::getXincrement() {
	return x_move;
}

int PointCloud::getYincrement() {
	return y_move;
}

int PointCloud::getZincrement() {
	return z_move;
}
void PointCloud::resetIncrements() {
	x_move = 0;
	y_move = 0;
	z_move = 0;
}

PointCloud::~PointCloud()
{
	// Delete the VBO and the VAO.
	// Failure to delete your VAOs, VBOs and other data given to OpenGL
	// is dangerous and may slow your program and cause memory leaks
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &vbon);
	glDeleteVertexArrays(1, &vao);
}

void PointCloud::draw()
{
	// Bind to the VAO.
	glBindVertexArray(vao);
	// Set point size.
	//glPointSize(pointSize);
	// Draw points 
    //glDrawArrays(GL_POINTS, 0, points.size());
	//glDrawArrays(GL_POINTS, 1, points.size());
	glDrawElements(GL_TRIANGLES, faces.size() , GL_UNSIGNED_INT, 0);
	// Unbind from the VAO.
	glBindVertexArray(0);
}

void PointCloud::update()
{
	// Spin the cube by 1 degree.
	// spin(1.0f);
	//spinCounter++;
	/*
	 * TODO: Section 3: Modify this function to spin the dragon and bunny about
	 * different axes. Look at the spin function for an idea
	 */
}

void PointCloud::updatePointSize(GLfloat size)
{
	/*
	 * TODO: Section 3: Implement this function to adjust the point size.
	 */
	pointSize = pointSize + size;
}

void PointCloud::spin(float deg)
{
	// Update the model matrix by multiplying a rotation matrix
	
	model = glm::rotate(model, glm::radians(deg), glm::vec3(0.0f, 1.0f, 0.0f));
	
	
}

