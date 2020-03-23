#include "Geometry.h"

Geometry::Geometry(std::string filename, GLuint modelLoc) {
	FILE* fp;

	model = glm::mat4(1);
	modelLocation = modelLoc;
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
	unsigned int v1n = 0;
	unsigned int v2n = 0;
	unsigned int v3n = 0;

	unsigned int v4 = 0;

	char delim = '/';
	//float r, g, b;
	char c1 = 'a';
	char c2 = 'a';
	fp = fopen(filename.c_str(), "r");

	while (c1 != EOF) {
		c1 = fgetc(fp);
		if (c1 == 'v') {
			c2 = fgetc(fp);
			if (c2 == ' ') {
				fscanf(fp, "%f %f %f", &x, &y, &z);
				pointsRaw.push_back(glm::vec3(x, y, z));
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
				normalsRaw.push_back(glm::vec3(xn, yn, zn));
			}
		}
		else if (c1 == 'f') {
			c2 = fgetc(fp);
			if (c2 == ' ') {
				//fscanf(fp, "%d%c%d%c%d %d%c%d%c%d %d%c%d%c%d", &v1, &delim,  &v2, &delim, &v2, &v2, &delim, &v3, &delim, &v3, &v3, &delim, &v4, &delim, &v4);
				fscanf(fp, "%d%c%d%c%d %d%c%d%c%d %d%c%d%c%d", &v1, &delim,  &v4, &delim, &v1n, &v2, &delim, &v4, &delim, &v2n, &v3, &delim, &v4, &delim, &v3n);
				vIndices.push_back((int) v1 - 1);
				vIndices.push_back((int) v2 - 1);
				vIndices.push_back((int) v3 - 1);
				vnIndices.push_back((int)v1n - 1);
				vnIndices.push_back((int)v2n - 1);
				vnIndices.push_back((int)v3n - 1);
			}

		}

	}

	for (int i = 0; i < vIndices.size(); i++) {
		points.push_back(pointsRaw[vIndices[i]]);
		normals.push_back(normalsRaw[vnIndices[i]]);
		indices.push_back(i);
	}

	float adjustmentX = (maxX + minX) / 2.0f;
	float adjustmentY = (maxY + minY) / 2.0f;
	float adjustmentZ = (maxZ + minZ) / 2.0f;
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
	
	fclose(fp);
	
	
	
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
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), indices.data(), GL_STATIC_DRAW);
	// Unbind everything
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Geometry::draw(glm::mat4 C) {
	model = C;
	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, glm::value_ptr(model));
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, indices.size() , GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void Geometry::update(glm::vec3 translation, int dir) {
	
}
Geometry::~Geometry()
{
	
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
}