#include "Node.h"

class Geometry : public Node {
public:
	glm::mat4 model;
	std::vector<Node*> children;
	std::vector<glm::vec3> points;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec3> normalsRaw;
	std::vector<glm::vec3> pointsRaw;
	std::vector<unsigned int> indices;
	std::vector<unsigned int> vIndices;
	std::vector<unsigned int> vnIndices;
	void addChild(Node* child) {
		children.push_back(child);
	}

	void draw(glm::mat4 C);

	void update(glm::vec3 translation, int dir);
	~Geometry();

	Geometry(std::string filename, GLuint modelLoc);
	GLuint vao, vbo, vbon, ebo, modelLocation;
};