//
//  Terrain.h
//  CSE167_HW0
//
//  Created by Anthony Joseph Tran on 3/15/20.
//  Copyright © 2020 Anthony Joseph Tran. All rights reserved.
//

#ifndef Terrain_h
#define Terrain_h

#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

const GLint m_TERRAIN_LENGTH = 257; // 2^n + 1 = length of heightmap
const GLfloat m_TERRAIN_SCALE = 8.0f;

const GLint m_INIT_MIN_HEIGHT = 55; // minimum for corner heights
const GLint m_INIT_RAND_RANGE = 30; // degree of variation in the corners

const GLint m_INIT_MIN_HEIGHT_OCEAN = 5;
const GLint m_INIT_RAND_RANGE_OCEAN = 5;

const GLint m_NOISE_RANGE = 10;

const GLfloat m_MAP_HEIGHT_SCALE = 15.0f;
const GLfloat m_MAP_BLOCK_SCALE = 1.0f;


class Terrain {
public:
    GLint heightmap[m_TERRAIN_LENGTH][m_TERRAIN_LENGTH] = {0}; // always a Square terrain
    glm::mat4 model;
    
    GLuint vao;
    GLuint vboVertex;
    GLuint vboTex;
    GLuint vboNorm;
    GLuint eboIndex;
    GLuint texID;
    
    GLint primitiveRestartIndex;
    GLint numOfIndices;
    GLint minHeight;
    GLint maxHeight;
    
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> textureCoordinates;
    std::vector<glm::vec3> normals;
    std::vector<GLuint> indices;
    
    // ctor/dtor
    Terrain();
    ~Terrain();
    
    // draw
    void render();
    
    // heightmap to terrain gen
    void buildFromHeightmap();
    
    void setupPositions();
    void setupTextureCoords();
    void setupNormals();
    void setupIndexBuffer();
    
    // other helpers
    GLfloat getHeightAt(const glm::vec3& position);
    
    // diamond square algorithm
    void diamondSquare(GLint size);
    void diamondStep(GLint x, GLint z, GLint reach);
    void squareStep(GLint x, GLint z, GLint reach);
    
    inline static float random(GLint range);
};

#endif /* Terrain_h */
