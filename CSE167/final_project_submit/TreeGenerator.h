#pragma once

#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>
#include <stack>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
class TreeGenerator {
public:
    struct Rule {
        char a;
        std::string b;
    };

    GLuint vao, vbo;
    std::vector<Rule> rules;
    Rule rule1;
    Rule rule2;
    Rule rule3;
    std::string axiom;
    std::string sentence;
    std::vector<glm::vec3> pts;
    int l = 10.0f;
    glm::vec3 pt = glm::vec3(0.0f, 0.0f, 0.0f);
    std::stack<glm::vec4> s;
    float theta = glm::radians(25.0f);
    float rot = 0.0;
	void generate();
	void turtle();
    void iterate(int iterations);
    void draw();
    TreeGenerator(char a, std::string b, char aa, std::string bb, char aaa, std::string bbb, std::string ax, int numRules);
    
};
