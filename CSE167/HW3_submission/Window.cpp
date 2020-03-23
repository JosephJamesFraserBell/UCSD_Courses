#include "Window.h"

#include "glm/gtx/string_cast.hpp"
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

/*
 * Declare your variables below. Unnamed namespace is used here to avoid
 * declaring global or static variables.
 */
namespace
{
	int width, height;
	std::string windowTitle("GLFW Starter Project");
	int counter = 0;
	glm::vec3 lastpt = glm::vec3(0.0f);
	
	// Boolean switches
	bool glRasterize = true;
	glm::vec2 mouseDelta;
	glm::vec2 lPoint;
	glm::vec2 cPoint;
	Cube* cube;

	
	Geometry* body;
	Geometry* antenna;
	Geometry* eyeball;
	Geometry* head;
	Geometry* limb;
	
	Transform* bodyTransform;
	Transform* antennaTransformL;
	Transform* antennaTransformR;
	Transform* eyeballTransformL;
	Transform* eyeballTransformR;
	Transform* headTransform;
	Transform* armTransformL;
	Transform* armTransformR;
	Transform* legTransformL;
	Transform* legTransformR;
	Transform* robotTransform1;
	Transform* robotTransform2;
	Transform* robotTransform3;
	Transform* robotTransform4;
	Transform* robotTransform5;
	Transform* robotTransform6;
	Transform* robotTransform7;
	Transform* robotTransform8;
	Transform* robotTransform9;
	Transform* robotTransform10;
	Transform* robotTransform11;
	Transform* robotTransform12;
	Transform* robotTransform13;
	Transform* robotTransform14;
	Transform* robotTransform15;
	Transform* robotTransform16;
	Transform* robotTransform17;
	Transform* robotTransform18;
	Transform* robotTransform19;
	Transform* robotTransform20;
	Transform* robotTransform21;
	Transform* robotTransform22;
	Transform* robotTransform23;
	Transform* robotTransform24;
	Transform* robotTransform25;
	Transform* robotArmy;

	PointCloud* sky;
	Object* currentObj;
	int z_counter = 0;
	glm::vec3 eye(0, 0, 0); // Camera position.
	glm::vec3 center(0, 0, 1); // The point we are looking at.
	glm::vec3 up(0, 1, 0); // The up direction of the camera.
	glm::vec3 eye2(0, 0, 0); // Camera position.
	glm::vec3 center2(0, 0, 1); // The point we are looking at.
	glm::vec3 up2(0, 1, 0); // The up direction of the camera.
	float fovy = 60;
	float near = 1;
	float far = 10000;
	glm::mat4 view = glm::lookAt(eye, center, up); // View matrix, defined by eye, center and up.
	glm::mat4 view2 = glm::lookAt(eye2, center2, up2); // View matrix, defined by eye, center and up.
	glm::mat4 projection; // Projection matrix.
	int stateL = 0;
	int stateR = 0;
	int stateZ = 0;
	glm::vec3 lastPoint(0.0f);
	int modelRotate = 1;
	GLuint program; // The shader program id.
	GLuint program2;
	GLuint program3;
	GLuint projectionLoc; // Location of projection in shader.
	GLuint viewLoc; // Location of view in shader.
	GLuint modelLoc; // Location of model in shader.
	GLuint projectionLoc2; // Location of projection in shader.
	GLuint viewLoc2; // Location of view in shader.
	GLuint modelLoc2; // Location of model in shader.
	GLuint projectionLoc3; // Location of projection in shader.
	GLuint viewLoc3; // Location of view in shader.
	GLuint modelLoc3; // Location of model in shader.
	GLuint colorLoc3;
	BezierCurve* b1;
	BezierCurve* b2;
	BezierCurve* b3;
	BezierCurve* b4;
	BezierCurve* b5;
	std::vector<glm::vec3> allPoints;
};

bool Window::initializeProgram()
{
	// Create a shader program with a vertex shader and a fragment shader.
	program = LoadShaders("shaders/shader2.vert", "shaders/shader2.frag");
	program2 = LoadShaders("shaders/shader.vert", "shaders/shader.frag");
	program3 = LoadShaders("shaders/shader3.vert", "shaders/shader3.frag");
	// This shader program is for displaying your rasterizer results
	// DO NOT MODIFY THESE SHADER PROGRAMS


	// Check the shader programs.
	if (!program)
	{
		std::cerr << "Failed to initialize shader program" << std::endl;
		return false;
	}
	if (!program2)
	{
		std::cerr << "Failed to initialize shader program 2" << std::endl;
		return false;
	}
	if (!program3)
	{
		std::cerr << "Failed to initialize shader program 3" << std::endl;
		return false;
	}


	// Activate the shader program.
	glUseProgram(program);
	// Get the locations of uniform variables.
	projectionLoc = glGetUniformLocation(program, "projection");
	viewLoc = glGetUniformLocation(program, "view");
	modelLoc = glGetUniformLocation(program, "model");
	projectionLoc2 = glGetUniformLocation(program2, "projection");
	viewLoc2 = glGetUniformLocation(program2, "view");
	modelLoc2 = glGetUniformLocation(program2, "model");
	projectionLoc3 = glGetUniformLocation(program3, "projection");
	viewLoc3 = glGetUniformLocation(program3, "view");
	modelLoc3 = glGetUniformLocation(program3, "model");
	colorLoc3 = glGetUniformLocation(program3, "color");
	
	return true;
}

bool Window::initializeObjects()
{
	// Create a cube of size 5.
	//cube = new Cube(5.0f);
	// Create a point cloud consisting of cube vertices.
	glm::mat4 T = glm::mat4(1.0f);
	glm::mat4 control_pts1 = glm::mat4(1.0f);
	glm::mat4 control_pts2 = glm::mat4(1.0f);
	glm::mat4 control_pts3 = glm::mat4(1.0f);
	glm::mat4 control_pts4 = glm::mat4(1.0f);
	glm::mat4 control_pts5 = glm::mat4(1.0f);

	glm::vec4 p0 = glm::vec4(-15.0f, 10.0f, 60.0f, 0.0f);
	glm::vec4 p1 = glm::vec4(-11.0f, 17.0f, 55.0f, 0.0f);
	glm::vec4 p4 = glm::vec4(10.0f, 15.0f, 35.0f, 0.0f);
	glm::vec4 p5 = glm::vec4(0.0f, 2.0f, 35.0f, 0.0f);
	glm::vec4 p8 = glm::vec4(10.0f, 10.0f, 40.0f, 0.0f);
	glm::vec4 p9 = glm::vec4(20.0f, 2.0f, 43.0f, 0.0f);
	glm::vec4 p12 = glm::vec4(33.0f, 22.0f, 43.0f, 0.0f);
	glm::vec4 p13 = glm::vec4(25.0f, 2.0f, 43.0f, 0.0f);
	glm::vec4 p16 = glm::vec4(20.0f, 0.0f, 35.0f, 0.0f);
	glm::vec4 p17 = glm::vec4(5.0f, 2.0f, 55.0f, 0.0f);

	glm::vec4 p2 = p4 + p4 - p5;
	glm::vec4 p3 = p4;


	glm::vec4 p6 = p8 + p8 - p9;
	glm::vec4 p7 = p8;

	
	glm::vec4 p10 = p12 + p12 - p13;
	glm::vec4 p11 = p12;

	
	glm::vec4 p14 = p16 + p16 - p17;
	glm::vec4 p15 = p16;

	
	glm::vec4 p18 = p0 + p0 - p1;
	glm::vec4 p19 = p0;


		

	control_pts1[0] = p0;
	control_pts1[1] = p1;
	control_pts1[2] = p2;
	control_pts1[3] = p3;

	control_pts2[0] = p4;
	control_pts2[1] = p5;
	control_pts2[2] = p6;
	control_pts2[3] = p7;

	control_pts3[0] = p8;
	control_pts3[1] = p9;
	control_pts3[2] = p10;
	control_pts3[3] = p11;

	control_pts4[0] = p12;
	control_pts4[1] = p13;
	control_pts4[2] = p14;
	control_pts4[3] = p15;

	control_pts5[0] = p16;
	control_pts5[1] = p17;
	control_pts5[2] = p18;
	control_pts5[3] = p19;

	
	b1 = new BezierCurve(control_pts1);
	b2 = new BezierCurve(control_pts2);
	b3 = new BezierCurve(control_pts3);
	b4 = new BezierCurve(control_pts4);
	b5 = new BezierCurve(control_pts5);
	
	std::vector<glm::vec3> points1 = b1->points;
	std::vector<glm::vec3> points2 = b2->points;
	std::vector<glm::vec3> points3 = b3->points;
	std::vector<glm::vec3> points4 = b4->points;
	std::vector<glm::vec3> points5 = b5->points;
	allPoints = points1;
	allPoints.insert(allPoints.end(), points2.begin(), points2.end());
	allPoints.insert(allPoints.end(), points3.begin(), points3.end());
	allPoints.insert(allPoints.end(), points4.begin(), points4.end());
	allPoints.insert(allPoints.end(), points5.begin(), points5.end());

	std::cout << allPoints.size() << std::endl;
	body = new Geometry("body_s.obj", modelLoc);
	antenna = new Geometry("antenna_s.obj", modelLoc);
	limb = new Geometry("limb_s.obj", modelLoc);
	eyeball = new Geometry("eyeball_s.obj", modelLoc);
	head = new Geometry("head_s.obj", modelLoc);
	
	bodyTransform = new Transform(glm::translate(T, glm::vec3(0.0f, 0.0f, 0.0f)));
	headTransform = new Transform(glm::translate(T, glm::vec3(0.0f, 1.5f, 0.0f)));
	antennaTransformL = new Transform(glm::rotate(glm::scale(glm::translate(T, glm::vec3(-0.4f, 1.8f, 0.0f)), glm::vec3(0.5f, 0.5f, 0.5f)), 0.5f, glm::vec3(0.0f, 0.0f, 1.0f)));
	antennaTransformR = new Transform(glm::rotate(glm::scale(glm::translate(T, glm::vec3(0.4f, 1.8f, 0.0f)), glm::vec3(0.5f, 0.5f, 0.5f)), -0.5f, glm::vec3(0.0f, 0.0f, 1.0f)));
	eyeballTransformL = new Transform(glm::translate(T, glm::vec3(-0.3f, 1.5f, -1.0f)));
	eyeballTransformR = new Transform(glm::translate(T, glm::vec3(0.3f, 1.5f, -1.0f)));
	
	armTransformL = new Transform(glm::translate(T, glm::vec3(-1.2f, 0.0f, 0.0f)));
	armTransformR = new Transform(glm::translate(T, glm::vec3(1.2f, 0.0f, 0.0f)));
	legTransformL = new Transform(glm::translate(T, glm::vec3(-0.4f, -1.2f, 0.0f)));
	legTransformR = new Transform(glm::translate(T, glm::vec3(0.4f, -1.2f, 0.0f)));

	robotTransform1 = new Transform(glm::translate(T, glm::vec3(6.0f,0.0f,-4.0f)));
	robotTransform2 = new Transform(glm::translate(T, glm::vec3(3.0f, 0.0f, -4.0f)));
	robotTransform3 = new Transform(glm::translate(T, glm::vec3(0.0f, 0.0f, -4.0f)));
	robotTransform4 = new Transform(glm::translate(T, glm::vec3(-3.0f, 0.0f, -4.0f)));
	robotTransform5 = new Transform(glm::translate(T, glm::vec3(-6.0f, 0.0f, -4.0f)));

	robotTransform6 = new Transform(glm::translate(T, glm::vec3(6.0f, 0.0f, -2.0f)));
	robotTransform7 = new Transform(glm::translate(T, glm::vec3(3.0f, 0.0f, -2.0f)));
	robotTransform8 = new Transform(glm::translate(T, glm::vec3(0.0f, 0.0f, -2.0f)));
	robotTransform9 = new Transform(glm::translate(T, glm::vec3(-3.0f, 0.0f, -2.0f)));
	robotTransform10 = new Transform(glm::translate(T, glm::vec3(-6.0f, 0.0f, -2.0f)));

	robotTransform11 = new Transform(glm::translate(T, glm::vec3(6.0f, 0.0f, 0.0f)));
	robotTransform12 = new Transform(glm::translate(T, glm::vec3(3.0f, 0.0f, 0.0f)));
	robotTransform13 = new Transform(glm::translate(T, glm::vec3(0.0f, 0.0f, 0.0f)));
	robotTransform14 = new Transform(glm::translate(T, glm::vec3(-3.0f, 0.0f, 0.0f)));
	robotTransform15 = new Transform(glm::translate(T, glm::vec3(-6.0f, 0.0f, 0.0f)));

	robotTransform16 = new Transform(glm::translate(T, glm::vec3(6.0f, 0.0f, 2.0f)));
	robotTransform17 = new Transform(glm::translate(T, glm::vec3(3.0f, 0.0f, 2.0f)));
	robotTransform18 = new Transform(glm::translate(T, glm::vec3(0.0f, 0.0f, 2.0f)));
	robotTransform19 = new Transform(glm::translate(T, glm::vec3(-3.0f, 0.0f, 2.0f)));
	robotTransform20 = new Transform(glm::translate(T, glm::vec3(-6.0f, 0.0f, 2.0f)));

	robotTransform21 = new Transform(glm::translate(T, glm::vec3(6.0f, 0.0f, 4.0f)));
	robotTransform22 = new Transform(glm::translate(T, glm::vec3(3.0f, 0.0f, 4.0f)));
	robotTransform23 = new Transform(glm::translate(T, glm::vec3(0.0f, 0.0f, 4.0f)));
	robotTransform24 = new Transform(glm::translate(T, glm::vec3(-3.0f, 0.0f, 4.0f)));
	robotTransform25 = new Transform(glm::translate(T, glm::vec3(-6.0f, 0.0f, 4.0f)));

	robotArmy = new Transform(T);

	bodyTransform->addChild(body);
	antennaTransformL->addChild(antenna);
	antennaTransformR->addChild(antenna);
	eyeballTransformL->addChild(eyeball);
	eyeballTransformR->addChild(eyeball);
	headTransform->addChild(head);
	armTransformL->addChild(limb);
	armTransformR->addChild(limb);
	legTransformL->addChild(limb);
	legTransformR->addChild(limb);

	robotTransform1->addChild(bodyTransform);
	robotTransform1->addChild(antennaTransformL);
	robotTransform1->addChild(antennaTransformR);
	robotTransform1->addChild(eyeballTransformL);
	robotTransform1->addChild(eyeballTransformR);
	robotTransform1->addChild(headTransform);
	robotTransform1->addChild(armTransformL);
	robotTransform1->addChild(armTransformR);
	robotTransform1->addChild(legTransformL);
	robotTransform1->addChild(legTransformR);

	robotTransform2->addChild(bodyTransform);
	robotTransform2->addChild(antennaTransformL);
	robotTransform2->addChild(antennaTransformR);
	robotTransform2->addChild(eyeballTransformL);
	robotTransform2->addChild(eyeballTransformR);
	robotTransform2->addChild(headTransform);
	robotTransform2->addChild(armTransformL);
	robotTransform2->addChild(armTransformR);
	robotTransform2->addChild(legTransformL);
	robotTransform2->addChild(legTransformR);

	robotTransform3->addChild(bodyTransform);
	robotTransform3->addChild(antennaTransformL);
	robotTransform3->addChild(antennaTransformR);
	robotTransform3->addChild(eyeballTransformL);
	robotTransform3->addChild(eyeballTransformR);
	robotTransform3->addChild(headTransform);
	robotTransform3->addChild(armTransformL);
	robotTransform3->addChild(armTransformR);
	robotTransform3->addChild(legTransformL);
	robotTransform3->addChild(legTransformR);

	robotTransform4->addChild(bodyTransform);
	robotTransform4->addChild(antennaTransformL);
	robotTransform4->addChild(antennaTransformR);
	robotTransform4->addChild(eyeballTransformL);
	robotTransform4->addChild(eyeballTransformR);
	robotTransform4->addChild(headTransform);
	robotTransform4->addChild(armTransformL);
	robotTransform4->addChild(armTransformR);
	robotTransform4->addChild(legTransformL);
	robotTransform4->addChild(legTransformR);

	robotTransform5->addChild(bodyTransform);
	robotTransform5->addChild(antennaTransformL);
	robotTransform5->addChild(antennaTransformR);
	robotTransform5->addChild(eyeballTransformL);
	robotTransform5->addChild(eyeballTransformR);
	robotTransform5->addChild(headTransform);
	robotTransform5->addChild(armTransformL);
	robotTransform5->addChild(armTransformR);
	robotTransform5->addChild(legTransformL);
	robotTransform5->addChild(legTransformR);

	robotTransform6->addChild(bodyTransform);
	robotTransform6->addChild(antennaTransformL);
	robotTransform6->addChild(antennaTransformR);
	robotTransform6->addChild(eyeballTransformL);
	robotTransform6->addChild(eyeballTransformR);
	robotTransform6->addChild(headTransform);
	robotTransform6->addChild(armTransformL);
	robotTransform6->addChild(armTransformR);
	robotTransform6->addChild(legTransformL);
	robotTransform6->addChild(legTransformR);

	robotTransform7->addChild(bodyTransform);
	robotTransform7->addChild(antennaTransformL);
	robotTransform7->addChild(antennaTransformR);
	robotTransform7->addChild(eyeballTransformL);
	robotTransform7->addChild(eyeballTransformR);
	robotTransform7->addChild(headTransform);
	robotTransform7->addChild(armTransformL);
	robotTransform7->addChild(armTransformR);
	robotTransform7->addChild(legTransformL);
	robotTransform7->addChild(legTransformR);

	robotTransform8->addChild(bodyTransform);
	robotTransform8->addChild(antennaTransformL);
	robotTransform8->addChild(antennaTransformR);
	robotTransform8->addChild(eyeballTransformL);
	robotTransform8->addChild(eyeballTransformR);
	robotTransform8->addChild(headTransform);
	robotTransform8->addChild(armTransformL);
	robotTransform8->addChild(armTransformR);
	robotTransform8->addChild(legTransformL);
	robotTransform8->addChild(legTransformR);

	robotTransform9->addChild(bodyTransform);
	robotTransform9->addChild(antennaTransformL);
	robotTransform9->addChild(antennaTransformR);
	robotTransform9->addChild(eyeballTransformL);
	robotTransform9->addChild(eyeballTransformR);
	robotTransform9->addChild(headTransform);
	robotTransform9->addChild(armTransformL);
	robotTransform9->addChild(armTransformR);
	robotTransform9->addChild(legTransformL);
	robotTransform9->addChild(legTransformR);

	robotTransform10->addChild(bodyTransform);
	robotTransform10->addChild(antennaTransformL);
	robotTransform10->addChild(antennaTransformR);
	robotTransform10->addChild(eyeballTransformL);
	robotTransform10->addChild(eyeballTransformR);
	robotTransform10->addChild(headTransform);
	robotTransform10->addChild(armTransformL);
	robotTransform10->addChild(armTransformR);
	robotTransform10->addChild(legTransformL);
	robotTransform10->addChild(legTransformR);

	robotTransform11->addChild(bodyTransform);
	robotTransform11->addChild(antennaTransformL);
	robotTransform11->addChild(antennaTransformR);
	robotTransform11->addChild(eyeballTransformL);
	robotTransform11->addChild(eyeballTransformR);
	robotTransform11->addChild(headTransform);
	robotTransform11->addChild(armTransformL);
	robotTransform11->addChild(armTransformR);
	robotTransform11->addChild(legTransformL);
	robotTransform11->addChild(legTransformR);

	robotTransform12->addChild(bodyTransform);
	robotTransform12->addChild(antennaTransformL);
	robotTransform12->addChild(antennaTransformR);
	robotTransform12->addChild(eyeballTransformL);
	robotTransform12->addChild(eyeballTransformR);
	robotTransform12->addChild(headTransform);
	robotTransform12->addChild(armTransformL);
	robotTransform12->addChild(armTransformR);
	robotTransform12->addChild(legTransformL);
	robotTransform12->addChild(legTransformR);

	robotTransform13->addChild(bodyTransform);
	robotTransform13->addChild(antennaTransformL);
	robotTransform13->addChild(antennaTransformR);
	robotTransform13->addChild(eyeballTransformL);
	robotTransform13->addChild(eyeballTransformR);
	robotTransform13->addChild(headTransform);
	robotTransform13->addChild(armTransformL);
	robotTransform13->addChild(armTransformR);
	robotTransform13->addChild(legTransformL);
	robotTransform13->addChild(legTransformR);

	robotTransform14->addChild(bodyTransform);
	robotTransform14->addChild(antennaTransformL);
	robotTransform14->addChild(antennaTransformR);
	robotTransform14->addChild(eyeballTransformL);
	robotTransform14->addChild(eyeballTransformR);
	robotTransform14->addChild(headTransform);
	robotTransform14->addChild(armTransformL);
	robotTransform14->addChild(armTransformR);
	robotTransform14->addChild(legTransformL);
	robotTransform14->addChild(legTransformR);

	robotTransform15->addChild(bodyTransform);
	robotTransform15->addChild(antennaTransformL);
	robotTransform15->addChild(antennaTransformR);
	robotTransform15->addChild(eyeballTransformL);
	robotTransform15->addChild(eyeballTransformR);
	robotTransform15->addChild(headTransform);
	robotTransform15->addChild(armTransformL);
	robotTransform15->addChild(armTransformR);
	robotTransform15->addChild(legTransformL);
	robotTransform15->addChild(legTransformR);

	robotTransform16->addChild(bodyTransform);
	robotTransform16->addChild(antennaTransformL);
	robotTransform16->addChild(antennaTransformR);
	robotTransform16->addChild(eyeballTransformL);
	robotTransform16->addChild(eyeballTransformR);
	robotTransform16->addChild(headTransform);
	robotTransform16->addChild(armTransformL);
	robotTransform16->addChild(armTransformR);
	robotTransform16->addChild(legTransformL);
	robotTransform16->addChild(legTransformR);

	robotTransform17->addChild(bodyTransform);
	robotTransform17->addChild(antennaTransformL);
	robotTransform17->addChild(antennaTransformR);
	robotTransform17->addChild(eyeballTransformL);
	robotTransform17->addChild(eyeballTransformR);
	robotTransform17->addChild(headTransform);
	robotTransform17->addChild(armTransformL);
	robotTransform17->addChild(armTransformR);
	robotTransform17->addChild(legTransformL);
	robotTransform17->addChild(legTransformR);

	robotTransform18->addChild(bodyTransform);
	robotTransform18->addChild(antennaTransformL);
	robotTransform18->addChild(antennaTransformR);
	robotTransform18->addChild(eyeballTransformL);
	robotTransform18->addChild(eyeballTransformR);
	robotTransform18->addChild(headTransform);
	robotTransform18->addChild(armTransformL);
	robotTransform18->addChild(armTransformR);
	robotTransform18->addChild(legTransformL);
	robotTransform18->addChild(legTransformR);

	robotTransform19->addChild(bodyTransform);
	robotTransform19->addChild(antennaTransformL);
	robotTransform19->addChild(antennaTransformR);
	robotTransform19->addChild(eyeballTransformL);
	robotTransform19->addChild(eyeballTransformR);
	robotTransform19->addChild(headTransform);
	robotTransform19->addChild(armTransformL);
	robotTransform19->addChild(armTransformR);
	robotTransform19->addChild(legTransformL);
	robotTransform19->addChild(legTransformR);

	robotTransform20->addChild(bodyTransform);
	robotTransform20->addChild(antennaTransformL);
	robotTransform20->addChild(antennaTransformR);
	robotTransform20->addChild(eyeballTransformL);
	robotTransform20->addChild(eyeballTransformR);
	robotTransform20->addChild(headTransform);
	robotTransform20->addChild(armTransformL);
	robotTransform20->addChild(armTransformR);
	robotTransform20->addChild(legTransformL);
	robotTransform20->addChild(legTransformR);

	robotTransform21->addChild(bodyTransform);
	robotTransform21->addChild(antennaTransformL);
	robotTransform21->addChild(antennaTransformR);
	robotTransform21->addChild(eyeballTransformL);
	robotTransform21->addChild(eyeballTransformR);
	robotTransform21->addChild(headTransform);
	robotTransform21->addChild(armTransformL);
	robotTransform21->addChild(armTransformR);
	robotTransform21->addChild(legTransformL);
	robotTransform21->addChild(legTransformR);

	robotTransform22->addChild(bodyTransform);
	robotTransform22->addChild(antennaTransformL);
	robotTransform22->addChild(antennaTransformR);
	robotTransform22->addChild(eyeballTransformL);
	robotTransform22->addChild(eyeballTransformR);
	robotTransform22->addChild(headTransform);
	robotTransform22->addChild(armTransformL);
	robotTransform22->addChild(armTransformR);
	robotTransform22->addChild(legTransformL);
	robotTransform22->addChild(legTransformR);

	robotTransform23->addChild(bodyTransform);
	robotTransform23->addChild(antennaTransformL);
	robotTransform23->addChild(antennaTransformR);
	robotTransform23->addChild(eyeballTransformL);
	robotTransform23->addChild(eyeballTransformR);
	robotTransform23->addChild(headTransform);
	robotTransform23->addChild(armTransformL);
	robotTransform23->addChild(armTransformR);
	robotTransform23->addChild(legTransformL);
	robotTransform23->addChild(legTransformR);

	robotTransform24->addChild(bodyTransform);
	robotTransform24->addChild(antennaTransformL);
	robotTransform24->addChild(antennaTransformR);
	robotTransform24->addChild(eyeballTransformL);
	robotTransform24->addChild(eyeballTransformR);
	robotTransform24->addChild(headTransform);
	robotTransform24->addChild(armTransformL);
	robotTransform24->addChild(armTransformR);
	robotTransform24->addChild(legTransformL);
	robotTransform24->addChild(legTransformR);

	robotTransform25->addChild(bodyTransform);
	robotTransform25->addChild(antennaTransformL);
	robotTransform25->addChild(antennaTransformR);
	robotTransform25->addChild(eyeballTransformL);
	robotTransform25->addChild(eyeballTransformR);
	robotTransform25->addChild(headTransform);
	robotTransform25->addChild(armTransformL);
	robotTransform25->addChild(armTransformR);
	robotTransform25->addChild(legTransformL);
	robotTransform25->addChild(legTransformR);



	robotArmy->addChild(robotTransform1);
	robotArmy->addChild(robotTransform2);
	robotArmy->addChild(robotTransform3);
	robotArmy->addChild(robotTransform4);
	robotArmy->addChild(robotTransform5);
	robotArmy->addChild(robotTransform6);
	robotArmy->addChild(robotTransform7);
	robotArmy->addChild(robotTransform8);
	robotArmy->addChild(robotTransform9);
	robotArmy->addChild(robotTransform10);
	robotArmy->addChild(robotTransform11);
	robotArmy->addChild(robotTransform12);
	robotArmy->addChild(robotTransform13);
	robotArmy->addChild(robotTransform14);
	robotArmy->addChild(robotTransform15);
	robotArmy->addChild(robotTransform16);
	robotArmy->addChild(robotTransform17);
	robotArmy->addChild(robotTransform18);
	robotArmy->addChild(robotTransform19);
	robotArmy->addChild(robotTransform20);
	robotArmy->addChild(robotTransform21);
	robotArmy->addChild(robotTransform22);
	robotArmy->addChild(robotTransform23);
	robotArmy->addChild(robotTransform24);
	robotArmy->addChild(robotTransform25);


	sky = new PointCloud("foo", 10);
	currentObj = sky;
	
	return true;
}

void Window::cleanUp()
{
	// Deallcoate the objects.
	delete body;
	delete antenna;
	delete eyeball;
	delete head;
	delete limb;
	delete bodyTransform;
	delete antennaTransformL;
	delete antennaTransformR;
	delete eyeballTransformL;
	delete eyeballTransformR;
	delete headTransform;
	delete armTransformL;
	delete armTransformR;
	delete legTransformL;
	delete legTransformR;
	delete robotTransform1;
	delete robotTransform2;
	delete robotTransform3;
	delete robotTransform4;
	delete robotTransform5;
	delete robotTransform6;
	delete robotTransform7;
	delete robotTransform8;
	delete robotTransform9;
	delete robotTransform10;
	delete robotTransform11;
	delete robotTransform12;
	delete robotTransform13;
	delete robotTransform14;
	delete robotTransform15;
	delete robotTransform16;
	delete robotTransform17;
	delete robotTransform18;
	delete robotTransform19;
	delete robotTransform20;
	delete robotTransform21;
	delete robotTransform22;
	delete robotTransform23;
	delete robotTransform24;
	delete robotTransform25;
	delete robotArmy;
	// Delete the shader programs.
	glDeleteProgram(program);
	glDeleteProgram(program2);
	glDeleteProgram(program3);

}

GLFWwindow* Window::createWindow(int width, int height)
{
	// Initialize GLFW.
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return NULL;
	}

	// 4x antialiasing.
	glfwWindowHint(GLFW_SAMPLES, 4);

#ifdef __APPLE__ 
	// Apple implements its own version of OpenGL and requires special treatments
	// to make it uses modern OpenGL.

	// Ensure that minimum OpenGL version is 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Enable forward compatibility and allow a modern OpenGL context
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// Create the GLFW window.
	GLFWwindow* window = glfwCreateWindow(width, height, windowTitle.c_str(), NULL, NULL);

	// Check if the window could not be created.
	if (!window)
	{
		std::cerr << "Failed to open GLFW window." << std::endl;
		glfwTerminate();
		return NULL;
	}

	// Make the context of the window.
	glfwMakeContextCurrent(window);

#ifndef __APPLE__
	// On Windows and Linux, we need GLEW to provide modern OpenGL functionality.

	// Initialize GLEW.
	if (glewInit())
	{
		std::cerr << "Failed to initialize GLEW" << std::endl;
		return NULL;
	}
#endif

	// Set swap interval to 1.
	glfwSwapInterval(0);

	// Initialize the quad that will be textured with your image
	// The quad must be made with the window

	// Call the resize callback to make sure things get drawn immediately.
	Window::resizeCallback(window, width, height);

	return window;
}

void Window::resizeCallback(GLFWwindow* window, int w, int h)
{
#ifdef __APPLE__
	// In case your Mac has a retina display.
	glfwGetFramebufferSize(window, &width, &height);
#endif
	width = w;
	height = h;

	// Resize our CPU rasterizer's pixel buffer and zbuffer

	// Set the viewport size.
	glViewport(0, 0, width, height);

	// Set the projection matrix.
	projection = glm::perspective(glm::radians(fovy),
		(float)width / (float)height, near, far);
}

void Window::idleCallback()
{

	armTransformR->update(glm::vec3(1.2f, 0.0f, 0.0f), 1);
	armTransformL->update(glm::vec3(-1.2f, 0.0f, 0.0f), -1);
	legTransformL->update(glm::vec3(-0.4f, -1.2f, 0.0f), -1);
	legTransformR->update(glm::vec3(0.4f, -1.2f, 0.0f), 1);
	


}

void Window::displayCallback(GLFWwindow* window)
{
	// Switch between OpenGL rasterizer and your rasterizer
	if (glRasterize) {

		
		glUseProgram(program2);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// Get the locations of uniform variables.
		glm::mat4 model2 = glm::mat4(1.0f);
		// Clear the color and depth buffers.
		glUniformMatrix4fv(projectionLoc2, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc2, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc2, 1, GL_FALSE, glm::value_ptr(model2));
		currentObj->draw();
		
		
		glUseProgram(program);
		

		glm::mat4 model = glm::mat4(1.0f);

		glm::vec3 pt = allPoints[counter];
		counter++;
		if (counter > 749){
			counter = 0;
		}

		glm::mat4 model23 = glm::translate(model, pt);
		lastpt = pt;
		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		robotArmy->draw(model23);
		
		
		glUseProgram(program3);
		glm::mat4 model5 = glm::scale(glm::mat4(1.0f) , glm::vec3(0.5f, 0.5f, 0.5f));
		//glEnable(GL_MAP1_VERTEX_3);
		glm::vec3 color = glm::vec3(1.0f, 0.0f, 0.0f);
		glUniformMatrix4fv(projectionLoc3, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc3, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc3, 1, GL_FALSE, glm::value_ptr(model5));
		glUniformMatrix4fv(colorLoc3, 1, GL_FALSE, glm::value_ptr(color));

		b1->draw();
		b2->draw();
		b3->draw();
		b4->draw();
		b5->draw();
		

		glfwPollEvents();
		glfwSwapBuffers(window);
		
	}
	
}

void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	/*
	 * TODO: Section 4: Modify below to add your key callbacks.
	 */

	 // Check for a key press.
	if (action == GLFW_PRESS)
	{
		glm::mat4 r1 = glm::mat4(1.0f);
		glm::mat4 model = robotArmy->M;
		glm::vec3 tran = glm::vec3(model[3]);
		// Uppercase key presses (shift held down + key press)
		if (mods == GLFW_MOD_SHIFT) {
			switch (key) {
			default:
				break;
			}
		}

		// Deals with lowercase key presses
		switch (key)
		{
		case GLFW_KEY_ESCAPE:
			// Close the window. This causes the program to also terminate.
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_1:
			// Set currentObj to cube
			break;
		case GLFW_KEY_2:
			// Set currentObj to cubePoints
			break;
		case GLFW_KEY_W:

			r1 = glm::rotate(r1, 0.2f, glm::vec3(1.0f, 0.0f, 0.0f));
			robotArmy->M = glm::translate(r1 * glm::translate(model, -tran), tran);
			break;
		case GLFW_KEY_A:
			

			r1 = glm::rotate(r1, -0.2f, glm::vec3(0.0f, 1.0f, 0.0f));
			robotArmy->M = glm::translate(r1 * glm::translate(model, -tran), tran);
			break;
		case GLFW_KEY_D:

			r1 = glm::rotate(r1, 0.2f, glm::vec3(0.0f, 1.0f, 0.0f));
			robotArmy->M = glm::translate(r1 * glm::translate(model, -tran), tran);
			break;
		case GLFW_KEY_S:

			r1 = glm::rotate(r1, -0.2f, glm::vec3(1.0f, 0.0f, 0.0f));
			robotArmy->M = glm::translate(r1 * glm::translate(model, -tran), tran);
			break;
		
		case GLFW_KEY_Z:

			r1 = glm::rotate(r1, -0.2f, glm::vec3(0.0f, 00.f, 1.0f));
			robotArmy->M = glm::translate(r1 * glm::translate(model, -tran), tran);
			break;
		case GLFW_KEY_X:

			r1 = glm::rotate(r1, 0.2f, glm::vec3(0.0f, 0.0f, 1.0f));
			robotArmy->M = glm::translate(r1 * glm::translate(model, -tran), tran);
			break;
		case GLFW_KEY_M:
			if (glRasterize)
				std::cout << "Switching to CPU rasterizer\n";
			else
				std::cout << "Switching to OpenGL rasterizer\n";
			glRasterize = !glRasterize;
			break;
		default:
			break;
		}
	}
}
glm::vec3 Window::trackBallMapping(double xposition, double yposition)
{
	glm::vec3 v = glm::vec3(0.0f);
	float d;
	v[0] = (2.0 * xposition - width) / width;
	v[1] = (height - 2.0 * yposition) / height;
	v[2] = 0.0;
	d = glm::length(v);
	d = (d < 1.0) ? d : 1.0;
	v[2] = sqrtf(1.001 - d * d);
	v = normalize(v); // Still need to normalize, since we only capped d, not v.
	std::cout << glm::to_string(v) << std::endl;
	return v;
}
void Window::mouseCallback(GLFWwindow* window, int button, int action, int mods) {
	double xpos = 0.0;
	double ypos = 0.0;
	if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
		stateL = 1;
		glfwGetCursorPos(window, &xpos, &ypos);
		//lastPoint = trackBallMapping(xpos, ypos);
		lPoint = glm::vec2(xpos, ypos);
	}

	else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT) {
		stateL = 0;
	}
	if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
		stateR = 1;
	}

	else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
		stateR = 0;
	}


}
void Window::cursorCallback(GLFWwindow* window, double xpos, double ypos) {
	glm::vec3 crossProduct;
	glm::mat4 current_model;

	if (stateL == 1) {

		cPoint = glm::vec2(xpos, ypos);
		mouseDelta = cPoint - lPoint;
		float velocity = glm::length(mouseDelta);
		float scale_rotate = 0.0001f;
		glm::mat4 rotate = glm::mat4(1.0f);
		if (velocity > 0.001) {

			center = glm::mat3(glm::rotate(-mouseDelta.x * scale_rotate, up)) * center;
			crossProduct = glm::cross(center, up);
			center = glm::mat3(glm::rotate(-mouseDelta.y * scale_rotate, crossProduct)) * center;

			lPoint - cPoint;

			view = glm::lookAt(eye, center, up); 

		}

	}

}

void Window::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	glm::mat4 zoom = glm::mat4(1.0f);
	glm::mat4 model = robotArmy->M;
	if (yoffset == 1) {
		robotArmy->M = glm::scale(robotArmy->M, glm::vec3(1.1f, 1.1f, 1.1f));
		z_counter++;
	}
	else if (yoffset == -1) {
		robotArmy->M = glm::scale(robotArmy->M, glm::vec3(0.9f, 0.9f, 0.9f));
		z_counter--;

	}

}


