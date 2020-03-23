#include "PointCloud.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
PointCloud::PointCloud(std::string objFilename, GLfloat pointSize)
	: pointSize(pointSize)
{
	
	 // Set the model matrix to an identity matrix. 
	model = glm::mat4(1);
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_STATIC_DRAW);
	
	// Enable vertex attribute 0. 
	// We will be able to access points through it.
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
	// Unbind from the VBO.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Unbind from the VAO.
	glBindVertexArray(0);
	cubemapTexture = loadCubemap(faces);
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
	

	
	glDepthFunc(GL_LEQUAL);
	glBindVertexArray(vao);
	
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	
	// Unbind from the VAO.
	glBindVertexArray(0);
	glDepthFunc(GL_LESS);
}
unsigned int PointCloud::loadCubemap(std::vector<std::string> faces)
{
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
	
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	int width, height, nrChannels;
	for (unsigned int i = 0; i < faces.size(); i++)
	{
		unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
				0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
			);
			stbi_image_free(data);
		}
		else
		{
			std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
			stbi_image_free(data);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return textureID;
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

void PointCloud::updatePointSize(GLfloat size)
{
	/*
	 * TODO: Section 3: Implement this function to adjust the point size.
	 */
}

void PointCloud::spin(float deg)
{
	// Update the model matrix by multiplying a rotation matrix
	model = glm::rotate(model, glm::radians(deg), glm::vec3(0.0f, 1.0f, 0.0f));
}

