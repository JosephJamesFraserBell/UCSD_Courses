//
//  Terrain.cpp
//  CSE167_HW0
//
//  Created by Anthony Joseph Tran on 3/15/20.
//  Copyright © 2020 Anthony Joseph Tran. All rights reserved.
//

#include "Terrain.h"
#include "stb_image.h"

Terrain::Terrain() {
    srand(static_cast<unsigned>(time(0)));
    model = glm::mat4(1.0f);
    
    // Scaling + translation
    model = glm::scale(model, glm::vec3(m_TERRAIN_SCALE));
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, -50.0f));
    
    // init corners w/ rand values
    heightmap[0][0] = m_INIT_MIN_HEIGHT + (rand() % m_INIT_RAND_RANGE); // top left
    heightmap[m_TERRAIN_LENGTH - 1][0] = m_INIT_MIN_HEIGHT + (rand() % m_INIT_RAND_RANGE); // bottom left

    // lower on this side for ocean
    heightmap[0][m_TERRAIN_LENGTH - 1] = m_INIT_MIN_HEIGHT_OCEAN + (rand() % m_INIT_RAND_RANGE_OCEAN); // top right
    heightmap[m_TERRAIN_LENGTH - 1][m_TERRAIN_LENGTH - 1] = m_INIT_MIN_HEIGHT_OCEAN + (rand() % m_INIT_RAND_RANGE_OCEAN); // bottom right

    diamondSquare(m_TERRAIN_LENGTH);
    
//    // Print heightmap
//    std::cout << '\n';
//
//    for (int i = 0; i < m_TERRAIN_LENGTH; ++i) {
//        for (int j = 0; j < m_TERRAIN_LENGTH; ++j) {
//            std::cout << heightmap[i][j] << " ";
//        }
//
//        std::cout << '\n';
//    }
//
//    std::cout << '\n';
    
    // Find max and min
    GLint max = INT_MIN, min = INT_MAX;
    for (int i = 0; i < m_TERRAIN_LENGTH; ++i) {
        for (int j = 0; j < m_TERRAIN_LENGTH; ++j) {
            if (heightmap[i][j] < min) {
                min = heightmap[i][j];
            } else if (heightmap[i][j] > max) {
                max = heightmap[i][j];
            }
        }
    }
    
    minHeight = min;
    maxHeight = max;
    
    // Populate fields and buffers
    buildFromHeightmap();
    
    // LOAD JPGs
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    
    int width, height, nrChannels;
    unsigned char *data = stbi_load("dry_sand.jpg", &width, &height, &nrChannels, 0);

    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "dry_sand.jpg texture failed to load!" << std::endl;
        std::cout << stbi_failure_reason() << std::endl;
    }
    stbi_image_free(data);
}

Terrain::~Terrain() {
    glDeleteVertexArrays(1, &vao);
    
    glDeleteBuffers(1, &vboVertex);
    glDeleteBuffers(1, &vboTex);
    glDeleteBuffers(1, &vboNorm);
    glDeleteBuffers(1, &eboIndex);
    
    glDeleteTextures(1, &texID);
}

void Terrain::render() {
    glBindVertexArray(vao);
    glBindTexture(GL_TEXTURE_2D, texID);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

// Helpers

void Terrain::buildFromHeightmap() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vboVertex);
    glGenBuffers(1, &vboTex);
    glGenBuffers(1, &vboNorm);
    glGenBuffers(1, &eboIndex);
    
    glBindVertexArray(vao);
    
    setupPositions();
    setupTextureCoords();
    setupIndexBuffer();
    setupNormals();
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

GLfloat Terrain::getHeightAt(const glm::vec3& position) {
    GLfloat height = -FLT_MAX;

    // Width and height of the terrain in world units
    GLfloat terrainSideLength = ( m_TERRAIN_LENGTH - 1) * m_MAP_BLOCK_SCALE;
    GLfloat halfTerrainLength = terrainSideLength * 0.5f;

    // Multiple the position by the inverse of the terrain matrix
    // to get the position in terrain local space

    // NOTE ** glm::mat4 must be change to inverseLocalToWorld if not creating at origin
    glm::vec3 terrainPos = glm::vec3(glm::inverse(model) * glm::vec4(position, 1.0f));
    glm::vec3 invBlockScale( 1.0f/m_MAP_BLOCK_SCALE, 0.0f, 1.0f/m_MAP_BLOCK_SCALE );

    // Calculate an offset and scale to get the vertex indices
    glm::vec3 offset(halfTerrainLength, 0.0f, halfTerrainLength);

    // Get the 4 vertices that make up the triangle we're over
    glm::vec3 vertexIndices = (terrainPos + offset) * invBlockScale;

    GLint u0 = (GLint) floorf(vertexIndices.x);
    GLint u1 = u0 + 1;
    GLint v0 = (GLint)floorf(vertexIndices.z);
    GLint v1 = v0 + 1;

    if (u0 >= 0 && u1 < m_TERRAIN_LENGTH && v0 >= 0 && v1 < m_TERRAIN_LENGTH)
    {
       glm::vec3 p00 = vertices[(v0 * m_TERRAIN_LENGTH) + u0];    // Top-left vertex
       glm::vec3 p10 = vertices[(v0 * m_TERRAIN_LENGTH) + u1];    // Top-right vertex
       glm::vec3 p01 = vertices[(v1 * m_TERRAIN_LENGTH) + u0];    // Bottom-left vertex
       glm::vec3 p11 = vertices[(v1 * m_TERRAIN_LENGTH) + u1];    // Bottom-right vertex

       // Determine which triangle pos is over
       GLfloat percentU = vertexIndices.x - u0;
       GLfloat percentV = vertexIndices.z - v0;

       glm::vec3 dU, dV;
       if (percentU > percentV)
       {   // Top triangle
           dU = p10 - p00;
           dV = p11 - p10;
       }
       else
       {   // Bottom triangle
           dU = p11 - p01;
           dV = p01 - p00;
       }

       glm::vec3 heightPos = p00 + (dU * percentU) + (dV * percentV);
       // Convert back to world-space by multiplying by the terrain's world matrix
        
        // NOTE ** change mat4 to inverseLocalWorldMat if not at origin
       heightPos = glm::vec3(model * glm::vec4(heightPos, 1));
       height = heightPos.y;
    }

    return height;
}

void Terrain::setupPositions() {
    vertices.resize(m_TERRAIN_LENGTH * m_TERRAIN_LENGTH);
    normals.resize(m_TERRAIN_LENGTH * m_TERRAIN_LENGTH);
    textureCoordinates.resize(m_TERRAIN_LENGTH * m_TERRAIN_LENGTH);
    
    // test
    GLfloat terrainSideLength = (m_TERRAIN_LENGTH - 1) * m_MAP_BLOCK_SCALE;
    GLfloat halfTerrainLength = terrainSideLength * 0.5f;

    for (int j = 0; j < m_TERRAIN_LENGTH; ++j) {
        for (int i = 0; i < m_TERRAIN_LENGTH; ++i) {
            GLuint index = (j * m_TERRAIN_LENGTH) + i;
            
            GLfloat S = (i / (GLfloat) (m_TERRAIN_LENGTH - 1));
            GLfloat T = (j / (GLfloat) (m_TERRAIN_LENGTH - 1));
            
            GLfloat X = (S * terrainSideLength) - halfTerrainLength;
            GLfloat Y = ((heightmap[i][j] - minHeight) / (GLfloat) (maxHeight - minHeight)) * m_MAP_HEIGHT_SCALE;
            GLfloat Z = (T * terrainSideLength) - halfTerrainLength;
            
            vertices[index] = glm::vec3(X, Y, Z);
            textureCoordinates[index] = glm::vec2(S, T);
            normals[index] = glm::vec3(0.0f);
        }
    }
    
//  -- print max and min heights -- //
    printf("Max Height: %d \nMin Height: %d\n", maxHeight, minHeight);

    
////  -- print vertices -- //
//    for (int i = 0; i < vertices.size(); ++i) {
//        std::cout << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << std::endl;
//    }
            
    glBindBuffer(GL_ARRAY_BUFFER, vboVertex);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
}

void Terrain::setupTextureCoords() {
    glBindBuffer(GL_ARRAY_BUFFER, vboTex);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * textureCoordinates.size(), textureCoordinates.data(), GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
}

void Terrain::setupNormals() {
    for (int i = 0; i < indices.size(); i += 3) {
        glm::vec3 v0 = vertices[indices[i]];
        glm::vec3 v1 = vertices[indices[i + 1]];
        glm::vec3 v2 = vertices[indices[i + 2]];
 
        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
 
        normals[indices[i]] += normal;
        normals[indices[i + 1]] += normal;
        normals[indices[i + 2]] += normal;
    }
     
    for (int i = 0; i < normals.size(); ++i) {
        normals[i] = glm::normalize(normals[i]);
    }

    glBindBuffer(GL_ARRAY_BUFFER, vboNorm);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normals.size(), normals.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
}
    

void Terrain::setupIndexBuffer() {
    // 2 triangles per quad, 3 indices per triangle
    const GLuint numTriangles = (m_TERRAIN_LENGTH - 1) * (m_TERRAIN_LENGTH - 1) * 2;
    indices.resize(numTriangles * 3);
    
    GLuint index = 0;
    for (GLuint j = 0; j < (m_TERRAIN_LENGTH - 1); ++j) {
        for (GLuint i = 0; i < (m_TERRAIN_LENGTH - 1); ++i) {
            GLuint vertexIndex = (j * m_TERRAIN_LENGTH) + i;
            
            // top triangle (T0)
            indices[index++] = (GLuint) (vertexIndex);                         // v0
            indices[index++] = (GLuint) (vertexIndex + m_TERRAIN_LENGTH + 1);  // v3
            indices[index++] = (GLuint) (vertexIndex + 1);                     // v1
            
            // bottom triangle (T1)
            indices[index++] = (GLuint) (vertexIndex);                         // v0
            indices[index++] = (GLuint) (vertexIndex + m_TERRAIN_LENGTH);      // v2
            indices[index++] = (GLuint) (vertexIndex + m_TERRAIN_LENGTH + 1);  // v3
        }
    }
    
    // Send indices to GPU
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboIndex);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_STATIC_DRAW);
    
////  --- print indices ---
//    for (int i = 0; i < indices.size(); i += 3) {
//        std::cout << indices[i] << " " << indices[i + 1] << " " << indices[i + 2] << std::endl;
//    }
}


//              DIAMOND SQUARE ALGORITHM HELPERS             //

// main algo
void Terrain::diamondSquare(GLint size) {
    GLint halfLength = size / 2;
    
    // BASE CASE - length not large enough for algo
    if (halfLength < 1)
        return;
    
    // Square
    for (int z = halfLength; z < m_TERRAIN_LENGTH; z+=size) {
        for (int x = halfLength; x < m_TERRAIN_LENGTH; x+=size) {
            squareStep(x % m_TERRAIN_LENGTH, z % m_TERRAIN_LENGTH, halfLength);
        }
    }
    
    // Diamond
    int col = 0;
    for (int x = 0; x < m_TERRAIN_LENGTH; x += halfLength) {
        col++;
        
        // odd
        if (col % 2 == 1) {
            for (int z = halfLength; z < m_TERRAIN_LENGTH; z+=size) {
                diamondStep(x % m_TERRAIN_LENGTH, z % m_TERRAIN_LENGTH, halfLength);
            }
        } else { // even
            for (int z = 0; z < m_TERRAIN_LENGTH; z+=size) {
                diamondStep(x % m_TERRAIN_LENGTH, z % m_TERRAIN_LENGTH, halfLength);
            }
        }
    }
    
    diamondSquare(size / 2);
}

// recursive methods
void Terrain::squareStep(GLint x, GLint z, GLint reach) {
    GLint count = 0;
    GLfloat avg = 0.0f;
    
    if (x - reach >= 0 && z - reach >= 0) {
        avg += heightmap[x-reach][z-reach];
        count++;
    }
    
    if (x - reach >= 0 && z + reach < m_TERRAIN_LENGTH) {
        avg += heightmap[x-reach][z+reach];
        count++;
    }
    
    if (x + reach < m_TERRAIN_LENGTH && z - reach >= 0) {
        avg += heightmap[x+reach][z-reach];
        count++;
    }
    
    if (x + reach < m_TERRAIN_LENGTH && z + reach < m_TERRAIN_LENGTH) {
        avg += heightmap[x+reach][z+reach];
        count++;
    }
    
    avg += random(reach); // define own randomness
    avg /= count;
    
    heightmap[x][z] = (GLint) avg;
}

void Terrain::diamondStep(GLint x, GLint z, GLint reach) {
    GLint count = 0;
    GLfloat avg = 0.0f;
    
    if (x - reach >= 0) {
        avg += heightmap[x - reach][z];
        count++;
    }
    
    if (x + reach < m_TERRAIN_LENGTH) {
        avg += heightmap[x + reach][z];
        count++;
    }
    
    if (z - reach >= 0) {
        avg += heightmap[x][z - reach];
        count++;
    }
    
    if (z + reach < m_TERRAIN_LENGTH) {
        avg += heightmap[x][z + reach];
        count++;
    }
    
    avg += random(reach);
    avg /= count;
    
    heightmap[x][z] = (GLint) avg;
}

inline float Terrain::random(GLint range) {
    // option 1: range-based random
    return (rand() % (range * 2)) - range;
    
    // option 2: self-defined
//    return (rand() % (m_NOISE_RANGE * 2)) - m_NOISE_RANGE;
}
