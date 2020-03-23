#include "PointCloud.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
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
	float x = 0.0;
	float y = 0.0;
	float z = 0.0;
	float xn = 0.0;
	float yn = 0.0;
	float zn = 0.0;
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
			}
			else if (c2 == 'n') {
				fscanf(fp, "%f %f %f", &xn, &yn, &zn);
				normals.push_back(glm::vec3(xn, yn, zn));
			}
		}
	}

	fclose(fp);

	
	/*
	 * TODO: Section 4, you will need to normalize the object to fit in the
	 * screen.
	 */

	 // Set the model matrix to an identity matrix. 
	model = glm::mat4(1);
	glm::mat4 initialTranslation = glm::mat4(1.0f);
	if (objFilename == "bunny.obj") {
		initialTranslation = glm::translate(initialTranslation, glm::vec3(-1.0f, 0.0f, 0.0f));
	}
	else if (objFilename == "dragon.obj") {
		initialTranslation = glm::translate(initialTranslation, glm::vec3(1.0f, 0.0f, 0.0f));
	}
	
	model = initialTranslation * model;
	// Set the color. 
	color = glm::vec3(1, 0, 0);

	// Generate a vertex array (VAO) and a vertex buffer objects (VBO).
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	// Bind to the VAO.
	// This tells OpenGL which data it should be paying attention to
	glBindVertexArray(vao);

	// Bind to the first VBO. We will use it to store the points.
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Pass in the data.
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * points.size(),
		points.data(), GL_STATIC_DRAW);
	// Enable vertex attribute 0. 
	// We will be able to access points through it.
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);

	/*
	 * TODO: Section 2 and 3. 	 
	 * Following the above example but for vertex normals, 
	 * 1) Generate a new vertex bufferbuffer,
	 * 2) Bind it as a GL_ARRAY_BUFFER type, 
	 * 3) Pass in the data 
	 * 4) Enable the next attribute array (which is 1)
	 * 5) Tell it how to process each vertex using glVertexAttribPointer
	 */
	
	

	glGenBuffers(1, &vbon);

	// Bind to the VAO.
	// This tells OpenGL which data it should be paying attention to
	//glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbon);
	// Pass in the data.
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normals.size(),
		normals.data(), GL_STATIC_DRAW);
	
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);

	// Unbind from the VBO.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Unbind from the VAO.
	glBindVertexArray(0);


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
	glDeleteVertexArrays(1, &vao);
}

void PointCloud::draw()
{
	// Bind to the VAO.
	glBindVertexArray(vao);
	// Set point size.
	glPointSize(pointSize);
	// Draw points 
	glDrawArrays(GL_POINTS, 0, points.size());
	glDrawArrays(GL_POINTS, 1, points.size());
	// Unbind from the VAO.
	glBindVertexArray(0);
}

void PointCloud::update()
{
	// Spin the cube by 1 degree.
	spin(1.0f);
	spinCounter++;
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
	if (objectName == "bunny.obj") {
		model = glm::rotate(model, glm::radians(deg), glm::vec3(0.0f, 1.0f, 0.0f));
	}
	else if (objectName == "dragon.obj") {
		model = glm::rotate(model, glm::radians(deg), glm::vec3(1.0f, 0.0f, 0.0f));
	}
	
}

