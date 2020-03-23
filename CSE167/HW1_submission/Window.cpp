#include "Window.h"
#include "glm/gtx/string_cast.hpp"
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
	Object* currentObj;  // The object currently displaying.
	Object* secondObj;  // The second object to display
	RasterizerQuad* quad;  // Object textured with your rasterization results

	glm::vec3 eye(0, 0, 20); // Camera position.
	glm::vec3 center(0, 0, 0); // The point we are looking at.
	glm::vec3 up(0, 1, 0); // The up direction of the camera.
	float fovy = 60;
	float near = 1;
	float far = 1000;
	glm::mat4 view = glm::lookAt(eye, center, up); // View matrix, defined by eye, center and up.
	glm::mat4 projection; // Projection matrix.

	GLuint program; // The shader program id.
	GLuint projectionLoc; // Location of projection in shader.
	GLuint viewLoc; // Location of view in shader.
	GLuint modelLoc; // Location of model in shader.
	GLuint colorLoc; // Location of color in shader.

	GLuint programQuad;
};

bool Window::initializeProgram()
{
	// Create a shader program with a vertex shader and a fragment shader.
	program = LoadShaders("shaders/shader.vert", "shaders/shader.frag");
	// This shader program is for displaying your rasterizer results
	// DO NOT MODIFY THESE SHADER PROGRAMS
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
	//cubePoints = new PointCloud("foo", 10);
	bunnyPoints = new PointCloud("bunny.obj", 2);
	dragonPoints = new PointCloud("dragon.obj", 5);
	// Set cube to be the first to display
	currentObj = bunnyPoints;
	secondObj = dragonPoints;
	return true;
}

void Window::cleanUp()
{
	// Deallcoate the objects.
	delete dragonPoints;
	delete bunnyPoints;
	delete quad;

	// Delete the shader programs.
	glDeleteProgram(program);
	glDeleteProgram(programQuad);
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
	secondObj->update();
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

		

		glm::mat4 modelBunny = currentObj->getModel();
		glm::vec3 colorBunny = currentObj->getColor();
		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelBunny));
		glUniform3fv(colorLoc, 1, glm::value_ptr(colorBunny));
		currentObj->draw();
		
		glm::mat4 modelDragon = secondObj->getModel();
		glm::vec3 colorDragon = secondObj->getColor();
		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelDragon));
		glUniform3fv(colorLoc, 1, glm::value_ptr(colorDragon));
		secondObj->draw();
	    
		
		
		
		// Gets events, including input such as keyboard and mouse or window resizing.
		glfwPollEvents();
		// Swap buffers.
		glfwSwapBuffers(window);
	}
	else {
		// Uncomment when you want to see your rasterizer results
		glUseProgram(programQuad);
		
		/*
		 * TODO: Section 5: Fill in clearBuffers() and rasterize() with your code. You should make sure to
		 * pass in the correct M, V, P, and D matrices to rasterize()
		 */
		quad->clearBuffers();
		//Do I need to modify these
		//glm::mat4 M = glm::mat4(1.0f);
		//glm::mat4 V = glm::mat4(1.0f);
		//glm::mat4 P = glm::mat4(1.0f);
		glm::mat4 D = glm::mat4(0.0f);

		glm::mat4 M = bunnyPoints->getModel();
		glm::mat4 V = view;
		glm::mat4 P = projection;


		glm::vec4 D0 = glm::vec4(0.0);
		glm::vec4 D1 = glm::vec4(0.0);;
		glm::vec4 D2 = glm::vec4(0.0);;
		glm::vec4 D3 = glm::vec4(0.0);;

		D0[0] = width/2;
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

		
		// CPU based rasterization. Fills in the pixel buffer and displays result. 
		// Replace cubePoints with the PointCloud of your choice
		quad->rasterize(M, V, P, D, *bunnyPoints);

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
		
		glm::mat4 model = currentObj->getModel();
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
				glm::mat4 translateReturn = glm::mat4(1.0f);
				int x_val = currentObj->x_move;
				int y_val = currentObj->y_move;
				int z_val = currentObj->z_move;
				translateReturn = glm::translate(translateReturn, glm::vec3(-1*x_val, -1*y_val, -1*z_val));
				model = translateReturn * model;
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
				currentObj->setModel(model);
				static_cast<PointCloud*>(currentObj)->resetIncrements();
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
			case GLFW_KEY_1:
				// Set currentObj to cube
				currentObj = bunnyPoints;
				secondObj = dragonPoints;
				break;
			case GLFW_KEY_2:
				// Set currentObj to cubePoints
				currentObj = dragonPoints;
				secondObj = bunnyPoints;
				break;
			case GLFW_KEY_M:
				if (glRasterize) {
					std::cout << "Switching to CPU rasterizer\n";
					
				}
					
				else {
					std::cout << "Switching to OpenGL rasterizer\n";
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
