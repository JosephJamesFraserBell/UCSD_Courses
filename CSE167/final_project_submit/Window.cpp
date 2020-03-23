#include "Window.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

/* 
 * Declare your variables below. Unnamed namespace is used here to avoid 
 * declaring global or static variables.
 */
namespace
{
    GLuint treeViewLoc;
    GLuint treeProjectionLoc;
    GLuint treeColorLoc;
    GLuint treeModelLoc;
    GLuint treeProgram;
    
    TreeGenerator* tree1; 
    TreeGenerator* tree2;
    TreeGenerator* tree3;

    glm::mat4 tree1Model;
    glm::mat4 tree2Model;
    glm::mat4 tree3Model;
    glm::mat4 particleModel;


    float tideAdjuster = 0.0f;
    float tideMove = 0.0f;
    float tideIncrement = 1.0f / 1000000.0f;
    float tideTime = 1.0f / 100000.0f;
	int width, height;
	std::string windowTitle("Beach Simulator");

    glm::vec3 oceanColor = glm::vec3(0.0f, 0.3f, 0.9f);
    glm::mat4 oceanModel = glm::mat4(1);
   
    std::vector<Window::Particle> particles;
    GLuint lastUsedParticle = 0;
    GLuint nr_particles = 2000;
    GLuint nr_new_particles = 2;
    // Cursor position
    glm::vec2 lastPos;
    GLboolean firstMouse = true;

	Object* currentObj; // The object currently displaying.
    PointCloud* ocean;

	glm::vec3 eye(40.0f, 100.0f, 0.1f); // Camera position.
	glm::vec3 center(0.0f, 0.0f, -1.0f); // The point we are looking at.
	glm::vec3 up(0.0f, 1.0f, 0.0f); // The up direction of the camera.

    unsigned int oceanTexture;

	float fovy = 60.0f;
	float near = 1.0f;
	float far = 10000.0f;

	glm::mat4 view = glm::lookAt(eye, center, up); // View matrix, defined by eye, center and up.
	glm::mat4 projection; // Projection matrix.

	GLuint program; // The shader program id.
    GLuint skyboxProgram; // Skybox shader program id
    GLuint particleProgram; // particle shader program id
    GLuint palmProgram;
    GLuint groundProgram; // Ground shader program id

    // Terrain
    Terrain* terrain;
    GLuint projectionGroundLoc;
    GLuint viewGroundLoc;
    GLuint modelGroundLoc;
    GLuint sandFlagLoc;

    // Normal Shader Locs
	GLuint projectionLoc; // Location of projection in shader.
	GLuint viewLoc; // Location of view in shader.
	GLuint modelLoc; // Location of model in shader.
	GLuint colorLoc; // Location of color in shader.
    GLuint timeLoc; // Location of time in shader.
    GLuint cameraPosLoc; // Location of camera position in shader.
    GLfloat waveTime = 0.0f;
    // Skybox PV Locs
    GLuint projectionSkyLoc; // Location of projection in shader.
    GLuint viewSkyLoc; // Location of view in shader.
    // particle Locs
    GLuint particleProjectionLoc;
    GLuint particleOffsetLoc;
    GLuint particleColorLoc;
    GLuint particleViewLoc;
    GLuint particleModelLoc;
    GLuint particleFlagLoc;
    //palm Locs
    GLuint palmProjectionLoc;
    GLuint palmModelLoc;
    GLuint palmViewLoc;
    GLuint palmColorLoc;

    PointCloud* palmTreeTrunk;
    PointCloud* palmTreeLeaves;
    Object* palm1;
    Object* palm2;
    // Skybox buffers
    GLuint skyboxVAO, skyboxVBO;
    GLuint particleVBO, particleVAO;
    // Skybox texture
    GLuint cubemapTexture;
    GLuint palmTrunkTexture;
    GLuint palmLeavesTexture;
    GLfloat dt = 1.0f;

    GLfloat oceanLevel = 0;


// ***      WINDOW FLAGS        *** //
    GLint sandFlag = 1;
// ***      WINDOW CONSTANTS     *** //
    
const GLfloat m_ROTSCALE = 0.002f;
const GLfloat m_MOVEMENTSCALE = 0.5f;
};

bool Window::initializeProgram()
{
	// Create ocean shader program with a vertex shader and a fragment shader.
	program = LoadShaders("shaders/shader.vert", "shaders/shader.frag");
    
    // Create skybox shader program with a vertex shader and a fragment shader.
    skyboxProgram = LoadShaders("shaders/skybox.vert", "shaders/skybox.frag");

    // Create particle shader program with a vertex shader and a fragment shader.
    particleProgram = LoadShaders("shaders/particle.vert", "shaders/particle.frag");

    palmProgram = LoadShaders("shaders/palm.vert", "shaders/palm.frag");
    
    // Create ground plane shader program
    groundProgram = LoadShaders("shaders/ground.vert", "shaders/ground.frag");
    
    treeProgram = LoadShaders("shaders/shaderTree.vert", "shaders/shaderTree.frag");

	// Check the shader program.
	if (!program)
	{
		std::cerr << "Failed to initialize shader program" << std::endl;
		return false;
	}
    
    if (!skyboxProgram)
    {
        std::cerr << "Failed to initialize skybox program" << std::endl;
        return false;
    }

    if (!particleProgram)
    {
        std::cerr << "Failed to initialize particle program" << std::endl;
        return false;
    }

    if (!palmProgram)
    {
        std::cerr << "Failed to initialize palm program" << std::endl;
        return false;
    }
    
    if (!groundProgram)
    {
        std::cerr << "Failed to initialize ground program" << std::endl;
        return false;
    }

    if (!groundProgram)
    {
        std::cerr << "Failed to initialize tree program" << std::endl;
        return false;
    }

	// Activate the shader program.
	glUseProgram(program);
	// Get the locations of uniform variables.
	projectionLoc = glGetUniformLocation(program, "projection");
	viewLoc = glGetUniformLocation(program, "view");
	modelLoc = glGetUniformLocation(program, "model");
	colorLoc = glGetUniformLocation(program, "color");
    timeLoc = glGetUniformLocation(program, "t");
    cameraPosLoc = glGetUniformLocation(program, "cameraPos");
    
    glUseProgram(skyboxProgram);
    // Get the locations of uniform variables.
    projectionSkyLoc = glGetUniformLocation(skyboxProgram, "projectionSky");
    viewSkyLoc = glGetUniformLocation(skyboxProgram, "viewSky");
    
    glUseProgram(particleProgram);
    particleProjectionLoc = glGetUniformLocation(particleProgram, "projection");
    particleColorLoc = glGetUniformLocation(particleProgram, "color");
    particleOffsetLoc = glGetUniformLocation(particleProgram, "offset");
    particleViewLoc = glGetUniformLocation(particleProgram, "view");
    particleModelLoc = glGetUniformLocation(particleProgram, "model");
    particleFlagLoc = glGetUniformLocation(particleProgram, "flag");

    glUseProgram(palmProgram);
    palmProjectionLoc = glGetUniformLocation(palmProgram, "projection");
    palmViewLoc = glGetUniformLocation(palmProgram, "view");
    palmModelLoc = glGetUniformLocation(palmProgram, "model");
    palmColorLoc = glGetUniformLocation(palmProgram, "color");

    glUseProgram(groundProgram);
    
    projectionGroundLoc = glGetUniformLocation(groundProgram, "projectionGround");
    viewGroundLoc = glGetUniformLocation(groundProgram, "viewGround");
    modelGroundLoc = glGetUniformLocation(groundProgram, "modelGround");
    sandFlagLoc = glGetUniformLocation(groundProgram, "sandFlag");

    glUseProgram(treeProgram);

    treeProjectionLoc = glGetUniformLocation(treeProgram, "projection");
    treeColorLoc = glGetUniformLocation(treeProgram, "color");
    treeViewLoc = glGetUniformLocation(treeProgram, "view");
    treeModelLoc = glGetUniformLocation(treeProgram, "model");

    srand(time(0));
    
	return true;
}

bool Window::initializeObjects()
{
   
    tree1 = new TreeGenerator('F', "FFX[FX[+XF]]", 'X', "FF[+XZ++X-F[+ZX]][-X++F-X]", 'Z', "[+F-X-F][++ZX]", "X", 3);
    tree2 = new TreeGenerator('F', "F[+F]F[-F][F]", 'aa', "bb", 'aaa', "bbb", "F", 1);
    tree3 = new TreeGenerator('F', "FFF[-FFF][+FFF]", 'X', "-F[+F][---X]+F-F[++++X]-X", 'aaa', "bbb", "F", 1);
    
    tree1Model = glm::mat4(1.0f);
    tree2Model = glm::mat4(1.0f);
    tree3Model = glm::mat4(1.0f);
    
    particleModel = glm::mat4(1.0f);

    tree1->iterate(5);
    tree2->iterate(5);
    tree3->iterate(3);
    for (GLuint i = 0; i < nr_particles; ++i) {
        particles.push_back(Particle());
    }
    std::cout << particles.size() << std::endl;

    std::cout << "Trunk" << std::endl;
    palmTreeTrunk = new PointCloud("MY_PALM.obj");
    palm1 = palmTreeTrunk;
    std::cout << "Leaves" << std::endl;
    palmTreeLeaves = new PointCloud("MY_PALM_LEAVES.obj");
    palm2 = palmTreeLeaves;

    glUseProgram(program);
    ocean = new PointCloud("ocean.obj");
    currentObj = ocean;
    oceanModel = glm::scale(oceanModel, glm::vec3(500.0f, 1.0f, 300.0f));
    oceanModel = glm::translate(oceanModel, glm::vec3(0.0f, 0.0f, 2.5f));

    int width, height, nrChannels;
    unsigned char* data = stbi_load("ocean5.jpg", &width, &height, &nrChannels, 0);
    
    glGenTextures(1, &oceanTexture);
    glBindTexture(GL_TEXTURE_2D, oceanTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    glUseProgram(skyboxProgram);
    
    // skybox VAO
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
    
    // Skybox textures
    cubemapTexture = loadCubemap(boxFaces);
    glBindVertexArray(0);

    glUseProgram(particleProgram);

    GLfloat particle_quad[] = {
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,

        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f
    };
    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);
    glBindVertexArray(particleVAO);
    // Fill mesh buffer
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(particle_quad), &particle_quad, GL_STATIC_DRAW);
    // Set mesh attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)0);
    glBindVertexArray(0);



    // Palm tree stuff
    glUseProgram(palmProgram);
    int width2, height2, nrChannels2;
    unsigned char* data2 = stbi_load("palmb3.jpg", &width2, &height2, &nrChannels2, 0);
    glGenTextures(1, &palmTrunkTexture);
    glBindTexture(GL_TEXTURE_2D, palmTrunkTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (data2)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width2, height2, 0, GL_RGB, GL_UNSIGNED_BYTE, data2);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data2);

    int width3, height3, nrChannels3;
    unsigned char* data3 = stbi_load("palml2.jpg", &width3, &height3, &nrChannels3, 0);

    glGenTextures(1, &palmLeavesTexture);
    glBindTexture(GL_TEXTURE_2D, palmLeavesTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (data3)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width3, height3, 0, GL_RGB, GL_UNSIGNED_BYTE, data3);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data3);

    
    glUseProgram(groundProgram);
    
    // Randomization and Translation
    terrain = new Terrain();
    
    oceanLevel = (terrain->minHeight + terrain->maxHeight) * 0.2f;
    oceanModel = glm::translate(oceanModel, glm::vec3(0.0f, (GLfloat) oceanLevel + 5.0f, 0.0f));
    
    // determine x and z of trees
    glm::vec3 tree1Pos = glm::vec3(rand() % 2000 - 1000, 0, rand() % 2000 - 1000);
    glm::vec3 tree2Pos = glm::vec3(rand() % 2000 - 1000, 0, rand() % 2000 - 1000);
    glm::vec3 tree3Pos = glm::vec3(rand() % 2000 - 1000, 0, rand() % 2000 - 1000);
    
    GLfloat tree1Y = terrain->getHeightAt(tree1Pos);
    GLfloat tree2Y = terrain->getHeightAt(tree2Pos);
    GLfloat tree3Y = terrain->getHeightAt(tree3Pos);
    
    tree1Model = glm::translate(tree1Model, glm::vec3(tree1Pos.x, tree1Y, tree1Pos.z));
    tree2Model = glm::translate(tree2Model, glm::vec3(tree2Pos.x, tree2Y, tree2Pos.z));
    tree3Model = glm::translate(tree3Model, glm::vec3(tree3Pos.x, tree3Y, tree3Pos.z));
    
    // palm tree
    GLfloat centerY = terrain->getHeightAt(glm::vec3(0.0f, 0.0f, 0.0f));
    palm1->model = glm::scale(palm1->model, glm::vec3(0.2f, 0.2f, 0.2f));
    palm1->model = glm::translate(palm1->model, glm::vec3(0.0f, centerY, 0.0f));
    
    // fire
    particleModel = glm::scale(particleModel, glm::vec3(1.5f, 1.5f, 1.5f));
    particleModel = glm::translate(particleModel, glm::vec3(5.0f, centerY, 3.0f));
    
	return true;
}

void Window::cleanUp()
{
	// Deallcoate the objects.
    delete ocean;
    delete terrain;

	// Delete the shader program.
	glDeleteProgram(program);
    glDeleteProgram(skyboxProgram);
    glDeleteProgram(particleProgram);
    glDeleteProgram(palmProgram);
    glDeleteProgram(groundProgram);
    glDeleteProgram(treeProgram);
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
	// Set the viewport size.
	glViewport(0, 0, width, height);

	// Set the projection matrix.
	projection = glm::perspective(glm::radians(fovy),
		(float)width / (float)height, near, far);
}

void Window::idleCallback()
{

    for (GLuint i = 0; i < nr_new_particles; ++i)
    {
        int unusedParticle = FirstUnusedParticle();
        //GLfloat random1 = ((rand() % 100) - 50) / 10.0f;
        //GLfloat random2 = ((rand() % 100) - 50) / 10.0f;
        RespawnParticle(particles[unusedParticle], glm::vec3(0.0f, -5.0f, -50.0f));
    }
    // Uupdate all particles
    for (GLuint i = 0; i < nr_particles; ++i)
    {
        Particle& p = particles[i];
        p.Life -= dt; // reduce life



        if (p.Life > 0.0f)
        {	// particle is alive, thus update
            float val = abs(((rand() % (int)2 * p.Velocity.y) - p.Velocity.y) / 2.0f);;
            p.Position.y += val;
            p.Position.x += ((rand() % (int)2.0f * p.Velocity.x) - 0.5f * p.Velocity.x) / 1.0f;
            p.Position.z += ((rand() % (int)2.0f * p.Velocity.x) - 0.5f * p.Velocity.x) / 1.0f;
            if (p.Life < 450.0f) {
                p.Position.y -= val * 0.5f;
            }
            if (p.Life < 300.0f) {
                p.Position.y -= val * 0.3f;
            }

            p.Color.a -= dt * 0.1f;
            p.Color.g -= dt * 0.005f;
            p.Color.b -= dt * 0.010f;

        }

    }
}

void Window::displayCallback(GLFWwindow* window)
{
    glUseProgram(palmProgram);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    drawPalmTree();


    glUseProgram(particleProgram);
    drawParticles();

    waveTime = waveTime + 1.0f;

    //std::cout << tideTime << std::endl;
    tideTime = tideTime + tideIncrement;

    if (tideTime > 0.0005) {
        tideIncrement = -1 * tideIncrement;
        tideAdjuster = 0.0005f;
    }
    else if (tideTime < 1.0f / 100000.0f) {
        tideIncrement = -1 * tideIncrement;
        tideAdjuster = 0.0f;
    }

    tideMove = tideTime - tideAdjuster;
    //std::cout << tideMove << std::endl;
    oceanModel = glm::translate(oceanModel, glm::vec3(0.0f, -tideMove * 100.0f, tideMove));
    glUseProgram(program);
    drawOcean();

    //Ground Shader Drawing
    glUseProgram(groundProgram);

    glUniformMatrix4fv(projectionGroundLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewGroundLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(modelGroundLoc, 1, GL_FALSE, glm::value_ptr(terrain->model));
    glUniform1i(sandFlagLoc, sandFlag);
    
    terrain->render();
    
    // Skybox Shader Drawing
    
    glDepthFunc(GL_LEQUAL);
    glUseProgram(skyboxProgram);

   glUniformMatrix4fv(projectionSkyLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewSkyLoc, 1, GL_FALSE, glm::value_ptr(view));

    glBindVertexArray(skyboxVAO);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // no padding on bytes

    glBindVertexArray(0);
    glDepthFunc(GL_LESS);
    
    //glm::vec3 color = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 color = glm::vec3(101.0f / 255.0f, 67.0f / 255.0f, 33.0f / 255.0f);
    glm::mat4 model = glm::mat4(1.0f);
    glUseProgram(treeProgram);
    glUniformMatrix4fv(treeProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(treeViewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(treeModelLoc, 1, GL_FALSE, glm::value_ptr(tree1Model));
    glUniform3fv(treeColorLoc, 1, glm::value_ptr(color));
    tree1->draw();
    
    glUniformMatrix4fv(treeModelLoc, 1, GL_FALSE, glm::value_ptr(tree2Model));
    tree2->draw();
    
    glUniformMatrix4fv(treeModelLoc, 1, GL_FALSE, glm::value_ptr(tree3Model));
    tree3->draw();

	// Gets events, including input such as keyboard and mouse or window resizing.
	glfwPollEvents();
	// Swap buffers.
	glfwSwapBuffers(window);
}

void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	 // Check for a key press.
	if (action == GLFW_PRESS)
	{
		switch (key)
		{
		case GLFW_KEY_ESCAPE:
			// Close the window. This causes the program to also terminate.
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
        case GLFW_KEY_1:
            sandFlag = !sandFlag;
            break;
        case GLFW_KEY_R:
            regenerateTerrain();
            break;
		default:
			break;
		}
	}
}

void Window::cursor_position_callback(GLFWwindow* window, GLdouble xpos, GLdouble ypos) {
    
    if (firstMouse) {
        lastPos = glm::vec2(xpos, ypos);
        firstMouse = false;
    }
    
    glm::vec2 currPos = glm::vec2(xpos, ypos);
    // Get the delta of cursor pos's and create direction vector
    glm::vec2 direction = currPos - lastPos;
    
    glm::vec3 toRotateAround = glm::cross(center, up);
    glm::mat4 xRotator = glm::rotate(-direction.x * m_ROTSCALE, up);
    glm::mat4 yRotator = glm::rotate(-direction.y * m_ROTSCALE, toRotateAround);
    
    // Create two rotators conditionally
    // xyRotator created and applied for bounds checking on center
    // If outside y bounds, only apply the x rotator and update cameraview mat to be as so
    glm::mat4 xyRotator = xRotator * yRotator;
    
    glm::vec3 tempCenterXY = glm::mat3(xyRotator) * center;
    glm::vec3 tempCenterX = glm::mat3(xRotator) * center;
        
    if (tempCenterXY.y > -0.90f && tempCenterXY.y < 0.90f) {
        view = glm::lookAt(eye, eye + tempCenterXY, up); // rotate on X and Y in normal bounds
        center = tempCenterXY;
    } else {
        view = glm::lookAt(eye, eye + tempCenterX, up); // out of y bounds, only rotate on X
        center = tempCenterX; // update center
    }
    
    // Update last pos for cursor delta
    lastPos = currPos;
}
// Helpers

void Window::move(Direction direction) {
    
    switch (direction) {
    case FORWARD:
        eye += m_MOVEMENTSCALE * center;
        view = glm::lookAt(eye, eye + center, up);
        break;
    case BACKWARD:
        eye -= m_MOVEMENTSCALE * center;
        view = glm::lookAt(eye, eye + center, up);
        break;
    case LEFT:
        eye -= glm::normalize(glm::cross(center,up)) * m_MOVEMENTSCALE;
        //eye += glm::vec3(-1.0, 0.0, 0.0) * m_MOVEMENTSCALE;
        view = glm::lookAt(eye, eye + center, up);
        break;
    case RIGHT:
        eye += glm::normalize(glm::cross(center, up)) * m_MOVEMENTSCALE;
        //eye -= glm::vec3(-1.0, 0.0, 0.0) * m_MOVEMENTSCALE;
        view = glm::lookAt(eye, eye + center, up);
        break;
    default:
        break;
    }
}

// Delete old terrain and create new Terrain object (for demo purposes)
void Window::regenerateTerrain() {
    delete terrain;
    terrain = new Terrain();
    
    oceanLevel = (terrain->minHeight + terrain->maxHeight) * 0.2f;
    
    oceanModel = glm::scale(glm::mat4(1.0f), glm::vec3(500.0f, 1.0f, 300.0f));
    oceanModel = glm::translate(oceanModel, glm::vec3(0.0f, oceanLevel + 5.0f, 2.5f));
    
    // trees
    // determine x and z of trees
    glm::vec3 tree1Pos = glm::vec3(rand() % 2000 - 1000, 0, rand() % 2000 - 1000);
    glm::vec3 tree2Pos = glm::vec3(rand() % 2000 - 1000, 0, rand() % 2000 - 1000);
    glm::vec3 tree3Pos = glm::vec3(rand() % 2000 - 1000, 0, rand() % 2000 - 1000);
    
    GLfloat tree1Y = terrain->getHeightAt(tree1Pos);
    GLfloat tree2Y = terrain->getHeightAt(tree2Pos);
    GLfloat tree3Y = terrain->getHeightAt(tree3Pos);
    
    tree1Model = glm::translate(glm::mat4(1.0f), glm::vec3(tree1Pos.x, tree1Y, tree1Pos.z));
    tree2Model = glm::translate(glm::mat4(1.0f), glm::vec3(tree2Pos.x, tree2Y, tree2Pos.z));
    tree3Model = glm::translate(glm::mat4(1.0f), glm::vec3(tree3Pos.x, tree3Y, tree3Pos.z));
    
    // palm tree
    GLfloat centerY = terrain->getHeightAt(glm::vec3(0.0f, 0.0f, 0.0f));
    palm1->model = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f, 0.2f, 0.2f));
    palm1->model = glm::translate(palm1->model, glm::vec3(0.0f, centerY, 0.0f));
    
    // fire
    particleModel = glm::scale(glm::mat4(1.0f), glm::vec3(1.5f, 1.5f, 1.5f));
    particleModel = glm::translate(particleModel, glm::vec3(5.0f, centerY, 3.0f));
}

// Not using texture.cpp, OpenGL version of cubemap loading
GLuint Window::loadCubemap(std::vector<std::string> boxFaces)
{
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (GLuint i = 0; i < boxFaces.size(); i++)
    {
        unsigned char *data = stbi_load(boxFaces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                         0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
            );
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << boxFaces[i] << std::endl;
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
GLuint Window::FirstUnusedParticle()
{
    // Search from last used particle, this will usually return almost instantly
    for (GLuint i = lastUsedParticle; i < nr_particles; ++i) {
        if (particles[i].Life <= 0.0f) {
            lastUsedParticle = i;
            return i;
        }
    }
    // Otherwise, do a linear search
    for (GLuint i = 0; i < lastUsedParticle; ++i) {
        if (particles[i].Life <= 0.0f) {
            lastUsedParticle = i;
            return i;
        }
    }
    // Override first particle if all others are alive
    lastUsedParticle = 0;
    return 0;
}
void Window::RespawnParticle(Particle& particle, glm::vec3 offset)
{
    GLfloat random1 = ((rand() % 100) - 50) / 50.0f;
    GLfloat random2 = ((rand() % 100) - 50) / 100.0f;
    GLfloat random3 = ((rand() % 100) - 50) / 100.0f;
    particle.Position = glm::vec3(random1, abs(random2), abs(random3)) + offset;

    particle.Color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    particle.Life = 500.0f;
    particle.Velocity = glm::vec3(1.0f, 1.0f, 1.0f);
}

void Window::drawParticles() {
    
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glUniform1i(particleFlagLoc, 1);
    for (Particle particle : particles)
    {
        if (particle.Life > 0.0f)
        {
            glUniform3fv(particleOffsetLoc, 1, glm::value_ptr(particle.Position));
            glUniform4fv(particleColorLoc, 1, glm::value_ptr(particle.Color));
            glUniformMatrix4fv(particleProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
            glUniformMatrix4fv(particleViewLoc, 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(particleModelLoc, 1, GL_FALSE, glm::value_ptr(particleModel));

            
            glBindVertexArray(particleVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
        }


    }
    glUniform1i(particleFlagLoc, 0);
    for (Particle particle : particles)
    {
        if (particle.Life > 0.0f)
        {
            glUniform3fv(particleOffsetLoc, 1, glm::value_ptr(particle.Position));
            glUniform4fv(particleColorLoc, 1, glm::value_ptr(particle.Color));
            glUniformMatrix4fv(particleProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
            glUniformMatrix4fv(particleViewLoc, 1, GL_FALSE, glm::value_ptr(view));

            glBindVertexArray(particleVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
        }


    }
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Window::drawOcean() {
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(oceanModel));
    glUniform3fv(colorLoc, 1, glm::value_ptr(oceanColor));
    glUniform3fv(cameraPosLoc, 1, glm::value_ptr(eye));
    glUniform1f(timeLoc, waveTime);
    glBindTexture(GL_TEXTURE_2D, oceanTexture);
    currentObj->draw();
}

void Window::drawPalmTree() {
    glUniformMatrix4fv(palmProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(palmViewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(palmModelLoc, 1, GL_FALSE, glm::value_ptr(palm1->getModel()));
    glUniform3fv(palmColorLoc, 1, glm::value_ptr(palm1->getColor()));

    glBindTexture(GL_TEXTURE_2D, palmTrunkTexture);
    palm1->draw();
    glBindTexture(GL_TEXTURE_2D, palmLeavesTexture);
    palm2->draw();
}


