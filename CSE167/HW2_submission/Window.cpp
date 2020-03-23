#include "Window.h"
#include "glm/gtx/string_cast.hpp"
#include "DirectionalLight.h"
#include "PointLight.h"
/* 
 * Declare your variables below. Unnamed namespace is used here to avoid 
 * declaring global or static variables.
 */
namespace
{
	int width, height;
	std::string windowTitle("GLFW Starter Project");

	// Boolean switches
	bool glRasterize = true;

	Cube* cube;
	PointCloud* cubePoints; 
	PointCloud* dragonPoints;
	PointCloud* bunnyPoints;
	PointCloud* bearPoints;
	PointCloud* spherePoints;
	Object* currentObj;  // The object currently displaying.
	Object* secondObj;  // The second object to display
	Object* thirdObj; // Thr third object to display
	Object* sphereObj; // Thr third object to display
	RasterizerQuad* quad;  // Object textured with your rasterization results

	DirectionalLight* lightSource;
	PointLight* pointLightSource;

	glm::vec3 eye(0, 0, 20); // Camera position.
	glm::vec3 center(0, 0, 0); // The point we are looking at.
	glm::vec3 up(0, 1, 0); // The up direction of the camera.
	float fovy = 60;
	float near = 1;
	float far = 1000;
	glm::mat4 view = glm::lookAt(eye, center, up); // View matrix, defined by eye, center and up.
	glm::mat4 projection; // Projection matrix.
	int stateL = 0;
	int stateR = 0;
	int stateZ = 0;
	glm::vec3 lastPoint(0.0f);
	int modelRotate = 1;
	GLuint program; // The shader program id.
	GLuint projectionLoc; // Location of projection in shader.
	GLuint viewLoc; // Location of view in shader.
	GLuint modelLoc; // Location of model in shader.
	GLuint colorLoc; // Location of color in shader.
	GLuint lightDirLoc; 
	GLuint pointLightLocationLoc;
	glm::vec3 pointLightLocation;
	GLuint lightColorLoc; 

	GLuint ambientLoc;
	GLuint diffuseLoc;
	GLuint specularLoc;

	GLuint programQuad;
	GLuint programPhong;

	glm::vec3 lightDir;
	glm::vec3 lightColor;
	GLfloat ambientCoefficient;
	GLfloat diffuseCoefficient;
	GLfloat specularCoefficient;

	int directionalLightFlag = 1;
	int pointLightFlag = 1;
	GLuint dlFlagLoc;
	GLuint plFlagLoc;
};

bool Window::initializeProgram()
{
	// Create a shader program with a vertex shader and a fragment shader.
	program = LoadShaders("shaders/shader.vert", "shaders/shader.frag");
	// This shader program is for displaying your rasterizer results
	// DO NOT MODIFY THESE SHADER PROGRAMS
	programPhong = LoadShaders("shaders/Phong.vert", "shaders/Phong.frag");
	programQuad = LoadShaders("shaders/RasterizerQuad.vert", "shaders/RasterizerQuad.frag");

	// Check the shader programs.
	if (!program)
	{
		std::cerr << "Failed to initialize shader program" << std::endl;
		return false;
	}
	if (!programQuad)
	{
		std::cerr << "Failed to initialize shader program" << std::endl;
		return false;
	}
	if (!programPhong)
	{
		std::cerr << "Failed to initialize Phong shader program" << std::endl;
		return false;
	}

	// Activate the shader program.
	glUseProgram(program);
	// Get the locations of uniform variables.
	projectionLoc = glGetUniformLocation(program, "projection");
	viewLoc = glGetUniformLocation(program, "view");
	modelLoc = glGetUniformLocation(program, "model");
	colorLoc = glGetUniformLocation(program, "color");

	return true;
}

bool Window::initializeObjects()
{
	// Create a cube of size 5.
	//cube = new Cube(5.0f);
	// Create a point cloud consisting of cube vertices.

	bunnyPoints = new PointCloud("bunny.obj", 2);
	dragonPoints = new PointCloud("dragon.obj", 4);
	bearPoints = new PointCloud("bear.obj", 5);
	spherePoints = new PointCloud("sphere.obj", 3);
	// Set cube to be the first to display
	currentObj = bunnyPoints;
	currentObj->setColor(glm::vec3(1, 0, 0));
	currentObj->ambientVal = 0.3f;
	currentObj->diffuseVal = 0.8f;
	currentObj->specularVal = 0.0f;

	secondObj = dragonPoints;
	secondObj->setColor(glm::vec3(0, 1, 0));
	secondObj->ambientVal = 0.3f;
	secondObj->diffuseVal = 0.0f;
	secondObj->specularVal = 1.0f;

	thirdObj = bearPoints;
	thirdObj->setColor(glm::vec3(0, 0, 1));
	thirdObj->ambientVal = 0.2f;
	thirdObj->diffuseVal = 0.7f;
	thirdObj->specularVal = 0.7f;

	sphereObj = spherePoints;
	glm::mat4 sphereModel = sphereObj->getModel();
	sphereModel = glm::scale(sphereModel, glm::vec3(0.1f, 0.1f, 0.1f));
	glm::mat4 translateRight = glm::mat4(1.0f);
	translateRight = glm::translate(translateRight, glm::vec3(1.0f, 0.0f, 0.0f));
	sphereModel = translateRight * sphereModel;
	sphereObj->setModel(sphereModel);
	sphereObj->ambientVal = 0.4f;
	sphereObj->diffuseVal = 0.0f;
	sphereObj->specularVal = 0.0f;

	lightSource = new DirectionalLight(glm::vec3(1.0f, 0.0f, 0.3f), glm::vec3(1,0,0));
	lightColor = lightSource->color;
	sphereObj->setColor(lightColor);
	lightDir = lightSource->direction;

	pointLightSource = new PointLight(glm::vec3(1.0f, 0.0f, 0.3f), glm::vec3(1.0f, 0.0f, 0.0f));
	sphereObj->setColor(lightColor);
	pointLightLocation = pointLightSource->location;

	return true;
}

void Window::cleanUp()
{
	// Deallcoate the objects.
	delete dragonPoints;
	delete bunnyPoints;
	delete bearPoints;
	delete quad;

	// Delete the shader programs.
	glDeleteProgram(program);
	glDeleteProgram(programQuad);
	glDeleteProgram(programPhong);
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
	quad = new RasterizerQuad(width, height);

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
	quad->updateBufSiz(width, height);

	// Set the viewport size.
	glViewport(0, 0, width, height);

	// Set the projection matrix.
	projection = glm::perspective(glm::radians(fovy),
		(float)width / (float)height, near, far);
}

void Window::idleCallback()
{
	// Perform any updates as necessary. 
	currentObj->update();
	glm::mat4 rotate = glm::mat4(1.0f);
	rotate = glm::rotate(rotate, glm::radians(0.8f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::vec4 lightDir2 = rotate * glm::vec4(lightDir, 1.0f);
	lightDir = glm::vec3(lightDir2[0], lightDir2[1], lightDir2[2]);
	

}

void Window::displayCallback(GLFWwindow* window)
{
	// Switch between OpenGL rasterizer and your rasterizer
	if (glRasterize) {
		// Switch back to using OpenGL's rasterizer
		glUseProgram(program);
		// Clear the color and depth buffers.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Specify the values of the uniform variables we are going to use.
		/*
		 * TODO: Section 3 and 4: Modify the code here to draw both the bunny and
		 * the dragon
		 * Note that the model matrix sent to the shader belongs only
		 * to what object the currentObj ptr is pointing to. You will need to
		 * use another call to glUniformMatrix4fv to change the model matrix
		 * data being sent to the vertex shader before you draw the other object
		 */

		

		glm::mat4 model = currentObj->getModel();
		glm::vec3 color = currentObj->getColor();
		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniform3fv(colorLoc, 1, glm::value_ptr(color));
		currentObj->draw();
		

	    
		
		
		
		// Gets events, including input such as keyboard and mouse or window resizing.
		glfwPollEvents();
		// Swap buffers.
		glfwSwapBuffers(window);
	}
	else {
		// Uncomment when you want to see your rasterizer results
		glUseProgram(programPhong);
		// Clear the color and depth buffers.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



		glm::mat4 model = currentObj->getModel();
		glm::vec3 color = currentObj->getColor();
		ambientCoefficient = currentObj->ambientVal;
		diffuseCoefficient = currentObj->diffuseVal;
		specularCoefficient = currentObj->specularVal;
		projectionLoc = glGetUniformLocation(programPhong, "projection");
		viewLoc = glGetUniformLocation(programPhong, "view");
		modelLoc = glGetUniformLocation(programPhong, "model");
		colorLoc = glGetUniformLocation(programPhong, "color");
		lightDirLoc = glGetUniformLocation(programPhong, "lightDir");
		lightColorLoc = glGetUniformLocation(programPhong, "lightColor");
		ambientLoc = glGetUniformLocation(programPhong, "ambientStrength");
		diffuseLoc = glGetUniformLocation(programPhong, "diffuseStrength");
		specularLoc = glGetUniformLocation(programPhong, "specularStrength");
		pointLightLocationLoc = glGetUniformLocation(programPhong, "plLocation");
		plFlagLoc = glGetUniformLocation(programPhong, "plFlag");
		dlFlagLoc = glGetUniformLocation(programPhong, "dlFlag");

		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniform3fv(colorLoc, 1, glm::value_ptr(color));
		glUniform3fv(lightDirLoc, 1, glm::value_ptr(lightDir));
		glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));
		glUniform1f(ambientLoc, ambientCoefficient);
		glUniform1f(diffuseLoc, diffuseCoefficient);
		glUniform1f(specularLoc, specularCoefficient);
		glUniform1i(plFlagLoc, pointLightFlag);
		glUniform1i(dlFlagLoc, directionalLightFlag);
		glUniform3fv(pointLightLocationLoc, 1, glm::value_ptr(pointLightLocation));
		currentObj->draw();

		model = sphereObj->getModel();
		color = sphereObj->getColor();
		
		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniform3fv(colorLoc, 1, glm::value_ptr(color));
		glUniform3fv(lightDirLoc, 1, glm::value_ptr(lightDir));
		glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));
		ambientCoefficient = sphereObj->ambientVal;
		diffuseCoefficient = sphereObj->diffuseVal;
		specularCoefficient = sphereObj->specularVal;
		glUniform1f(ambientLoc, ambientCoefficient);
		glUniform1f(diffuseLoc, diffuseCoefficient);
		glUniform1f(specularLoc, specularCoefficient);
		
		sphereObj->draw();


		// Gets events, including input such as keyboard and mouse or window resizing.
		glfwPollEvents();
		// Swap buffers.
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
		
		glm::mat4 model = currentObj->getModel();
		glm::mat4 lightModel = sphereObj->getModel();
		// Uppercase key presses (shift held down + key press)
		if (mods == GLFW_MOD_SHIFT) {
			
			switch (key) {
			case GLFW_KEY_P:
				if (glRasterize) {
					static_cast<PointCloud*>(currentObj)->updatePointSize(1);
				}
				
			
				if (!glRasterize) {
					bunnyPoints->n++;
				}
				
				
				
				break;
			case GLFW_KEY_C:
				glm::vec3 scaleUp = glm::vec3(1.1f, 1.1f, 1.1f);
				model = glm::scale(model, scaleUp);
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				currentObj->scaleFactor += 1;
				std::cout << currentObj->scaleFactor << std::endl;
				break;
			case GLFW_KEY_R: {
				int scale_val = currentObj->scaleFactor;
				glm::vec3 scaleUp = glm::vec3(1.1f, 1.1f, 1.1f);
				glm::vec3 scaleDown = glm::vec3(0.91f, 0.91f, 0.91f);

				if (currentObj->spinCounter > 0){

					
					if (currentObj->objectName == "bunny.obj") {
						model = glm::rotate(model, glm::radians(-1*currentObj->spinCounter), glm::vec3(0.0f, 1.0f, 0.0f));
						currentObj->spinCounter = 0;
					}
					else if (currentObj->objectName == "dragon.obj") {
						model = glm::rotate(model, glm::radians(-1 * currentObj->spinCounter), glm::vec3(1.0f, 0.0f, 0.0f));
						currentObj->spinCounter = 0;
					}
					
				}
				if (scale_val > 0) {

					for (int i = 0; i < scale_val; i++) {
						model = glm::scale(model, scaleDown);
						
					}
				}
				else if (scale_val < 0) {
					for (int i = 0; i < -1 * scale_val; i++) {
						model = glm::scale(model, scaleUp);
					}

				}

				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				currentObj->scaleFactor = 0;
				break;
			}
			case GLFW_KEY_Z:
				
				glm::mat4 translateForward = glm::mat4(1.0f);
				translateForward = glm::translate(translateForward, glm::vec3(0.0f, 0.0f, 1.0f));
				model = translateForward * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->incrementZ();
				break;
			default:
				break;
			}
		}
		else {
			switch (key)
			{
			case GLFW_KEY_ESCAPE:
				// Close the window. This causes the program to also terminate.
				glfwSetWindowShouldClose(window, GL_TRUE);
				break;
			case GLFW_KEY_0:
				modelRotate = !modelRotate;
			break;
			case GLFW_KEY_1:
				//only directional lighting
				directionalLightFlag = 1;
				pointLightFlag = 0;
				break;
			case GLFW_KEY_2:
				//directional lighting and point light
				directionalLightFlag = 1;
				pointLightFlag = 1;
				break;
			case GLFW_KEY_3:
				directionalLightFlag = !directionalLightFlag;
				break;
			case GLFW_KEY_P:
				if (glRasterize) {
					static_cast<PointCloud*>(currentObj)->updatePointSize(-1);
				}
				
				if (!glRasterize) {
					if (bunnyPoints->n > 1) {
						bunnyPoints->n--;
					}
				}
				break;
			case GLFW_KEY_R: {
				
				
				// Set currentObj to cube
				glm::mat4 mInv = inverse(model);
				model = mInv * model;
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->resetIncrements();

				mInv = inverse(lightModel);
				lightModel = mInv * lightModel;

				lightModel = glm::scale(lightModel, glm::vec3(0.1f, 0.1f, 0.1f));
				glm::mat4 translateRight = glm::mat4(1.0f);
				translateRight = glm::translate(translateRight, glm::vec3(1.0f, 0.0f, 0.0f));
				lightModel = translateRight * lightModel;

				sphereObj->setModel(lightModel);
				pointLightLocation = glm::vec3(1.0f, 0.0f, 0.0f);
				/*
				glm::mat4 translateReturn = glm::mat4(1.0f);
				int x_val = currentObj->x_move;
				int y_val = currentObj->y_move;
				int z_val = currentObj->z_move;
				translateReturn = glm::translate(translateReturn, glm::vec3(-1*x_val, -1*y_val, -1*z_val));
				model = translateReturn * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->resetIncrements();
				/**/
			}
				break;
			case GLFW_KEY_A:
				// Set currentObj to cube
				glm::mat4 translateLeft = glm::mat4(1.0f);
				translateLeft = glm::translate(translateLeft, glm::vec3(-1.0f, 0.0f, 0.0f));
				model = translateLeft * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->decrementX();
				break;
			case GLFW_KEY_D:
				glm::mat4 translateRight = glm::mat4(1.0f);
				translateRight = glm::translate(translateRight, glm::vec3(1.0f, 0.0f, 0.0f));
				model = translateRight * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->incrementX();
				break;
			case GLFW_KEY_W:
				// Set currentObj to cube
				glm::mat4 translateUp = glm::mat4(1.0f);
				translateUp = glm::translate(translateUp, glm::vec3(0.0f, 1.0f, 0.0f));
				model = translateUp * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->incrementY();
				break;
			case GLFW_KEY_S:
				glm::mat4 translateDown = glm::mat4(1.0f);
				translateDown = glm::translate(translateDown, glm::vec3(0.0f, -1.0f, 0.0f));
				model = translateDown * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->decrementY();
				break;
			case GLFW_KEY_Z:
				glm::mat4 translateBack = glm::mat4(1.0f);
				translateBack = glm::translate(translateBack, glm::vec3(0.0f, 0.0f, -1.0f));
				model = translateBack * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->decrementZ();
				break;
			case GLFW_KEY_C:
				glm::vec3 scaleDown = glm::vec3(0.91f, 0.91f, 0.91f);
				model = glm::scale(model, scaleDown);
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				currentObj->scaleFactor -= 1;
				std::cout << currentObj->scaleFactor << std::endl;
				break;
			case GLFW_KEY_F1:
				// Set currentObj to cube
				currentObj = bunnyPoints;
				lightSource = new DirectionalLight(glm::vec3(1.0f, 0.0f, 0.3f), glm::vec3(1, 0, 0));
				lightColor = lightSource->color;
				lightDir = lightSource->direction;
				

				break;
			case GLFW_KEY_F2:
				// Set currentObj to cubePoints
				currentObj = dragonPoints;
				lightSource = new DirectionalLight(glm::vec3(0.0f, 0.0f, 1.0f), lightDir);
				lightColor = lightSource->color;
				lightDir = lightSource->direction;
				
				
				break;
			case GLFW_KEY_F3:
				// Set currentObj to cubePoints
				currentObj = bearPoints;
				lightSource = new DirectionalLight(glm::vec3(0.0f, 1.0f, 0.2f), lightDir);
				lightColor = lightSource->color;
				lightDir = lightSource->direction;
				
				break;
			case GLFW_KEY_N:
				if (glRasterize) {
					std::cout << "Switching to phong model\n";
				}
					
				else {
					std::cout << "Switching to normal coloring\n";
				}
					
				glRasterize = !glRasterize;
				break;
			default:
				break;
			}
		}
		// Deals with lowercase key presses
		
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
		lastPoint = trackBallMapping(xpos, ypos);
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
	glm::vec3 currentPoint;
	glm::vec3 direction;
	glm::vec3 crossProduct;
	glm::mat4 model;
	
	float translateX;
	float translateY;
	float lastX = 0.0f;
	float lastY = 0.0f;
	glm::mat4 current_model;

	if (stateL == 1) {
		//std::cout << xpos << "," << ypos << std::endl;
		currentPoint = trackBallMapping(xpos, ypos);
		direction = currentPoint - lastPoint;
		float velocity = glm::length(direction);
		float scale_rotate = 5.0f;
		glm::mat4 rotate = glm::mat4(1.0f);
		if (velocity > 0.001) {
			
			crossProduct = normalize(glm::cross(lastPoint, currentPoint));
			std::cout << "crossProd: " << glm::to_string(crossProduct) << std::endl;
			float angle_rotate = velocity * scale_rotate;
			//std::cout << angle_rotate << std::endl;
			if (modelRotate) {
				current_model = currentObj->getModel();
			}
			else if (!modelRotate){
				current_model = sphereObj->getModel();
			}
			

			rotate = glm::rotate(rotate, glm::radians(angle_rotate), crossProduct); 
			model = rotate * current_model;
			if (modelRotate) {
				currentObj->setModel(model);
			}
			else if (!modelRotate) {
				sphereObj->setModel(model);
				pointLightLocation = glm::vec3(rotate * glm::vec4(pointLightLocation, 1.0f));
			}
			
		}

	}
	if (stateR == 1) {
		glm::mat4 D = glm::mat4(0.0f);

		glm::vec4 D0 = glm::vec4(0.0);
		glm::vec4 D1 = glm::vec4(0.0);;
		glm::vec4 D2 = glm::vec4(0.0);;
		glm::vec4 D3 = glm::vec4(0.0);;
		D0[0] = width / 2;
		D1[1] = height / 2;
		D2[2] = 0.5f;
		D3[0] = width / 2;
		D3[1] = height / 2;
		D3[2] = 0.5f;
		D3[3] = 1;
		D[0] = D0;
		D[1] = D1;
		D[2] = D2;
		D[3] = D3;

		model = currentObj->getModel();
		float z_component = model[3][2];
		glm::mat4 modelInv = inverse(model);
		model = modelInv * model;
		model[3][2] = z_component;
		currentObj->setModel(model);
		model = currentObj->getModel();
		glm::vec4 t = glm::vec4((float)xpos, (float)(height - ypos), 0.0f, 1.0f);
		t = inverse(D*projection*view) * t;
		float t_scale = t[3];
		t = t / t_scale;
		
		glm::mat4 translate = glm::mat4(1.0f);
		translate = glm::translate(translate, glm::vec3(t[0]*t[2], t[1]*t[2], 0));
		model = translate * model;
		currentObj->setModel(model);
	
	}

}
void Window::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	glm::mat4 model = currentObj->getModel();
	glm::mat4 zoom = glm::mat4(1.0f);
	glm::mat4 sphereModel = sphereObj->getModel();
	glm::vec3 scaleLight;
	if (yoffset == 1) {
		zoom = glm::translate(zoom, glm::vec3(0.0f, 0.0f, 1.0f));
		scaleLight = glm::vec3(-1.0f, -1.0f, -1.0f);
		static_cast<PointCloud*>(currentObj)->incrementZ();
	}
	else if (yoffset == -1) {
		zoom = glm::translate(zoom, glm::vec3(0.0f, 0.0f, -1.0f));
		scaleLight = glm::vec3(1.0f, 1.0f, 1.0f);
		static_cast<PointCloud*>(currentObj)->decrementZ();
	}
	
	model = zoom * model;
	pointLightLocation = glm::vec3(zoom * glm::vec4(pointLightLocation, 1.0f));
	currentObj->setModel(model);
	sphereModel = zoom * sphereModel;
	sphereModel = glm::scale(sphereModel, scaleLight);
	sphereObj->setModel(sphereModel);
	
	
}

