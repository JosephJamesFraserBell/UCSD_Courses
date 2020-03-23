#include "Node.h"
class Transform : public Node {
public:
	glm::mat4 M ;
	std::vector<Node*> children;

	void addChild(Node* child);

	void draw(glm::mat4 C);
	void update(glm::vec3 translation, int dir);

	Transform(glm::mat4 X);

	int counter = 0;
	int increaseFlag = 1;
	
	glm::mat4 T;
	glm::mat4 R;
	glm::mat4 modelInv;
	std::vector<Node*>::iterator citer;
};

