#include "Transform.h"

Transform::Transform(glm::mat4 X) {
	M = X;
}

void Transform::addChild(Node* child) {
	children.push_back(child);
}

void Transform::draw(glm::mat4 C) {

	
	for (citer = children.begin(); citer != children.end(); citer++) {
		(*citer)->draw(C*M);
	}
}

void Transform::update(glm::vec3 translation, int dir ) {
	
	if (counter > 25) {
		increaseFlag = 0;
	}
	else if (counter < -25) {
		increaseFlag = 1;
	}
	if (increaseFlag) {
		counter++;
	}
	if (!increaseFlag) {
		counter--;
	}
	
	T = glm::mat4(1.0f);
	R = glm::mat4(1.0f);
	

	
	T = glm::translate(T, translation);
	modelInv = inverse(M);
	
	R = glm::rotate(R, dir*counter*0.05f, glm::vec3(1.0f, 0.0f, 0.0f));
	M = T * R * modelInv * M;
}



