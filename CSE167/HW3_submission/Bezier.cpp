#include "Bezier.h"

void Bezier::addChild(Bezier* child){
	children.push_back(child);
}

void Bezier::draw() {
	for (citer = children.begin(); citer != children.end(); citer++) {
		(*citer)->draw();
	}
}

Bezier::Bezier() {
	std::cout << "Created Bezier Parent" << std::endl;
}