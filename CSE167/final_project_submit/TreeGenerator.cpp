#include "TreeGenerator.h"

TreeGenerator::TreeGenerator(char a, std::string b, char aa, std::string bb, char aaa, std::string bbb, std::string ax, int numRules) {
    if (numRules == 3) {
        rule1 = { a, b };
        rule2 = { aa, bb };
        rule3 = { aaa, bbb };
        rules.push_back(rule1);
        rules.push_back(rule2);
        rules.push_back(rule3);
    }
    else if (numRules == 2) {
        rule1 = { a, b };
        rule2 = { aa, bb };
        rules.push_back(rule1);
        rules.push_back(rule2);
        
    }
    else if (numRules == 1) {
        rule1 = { a, b }; 
        rules.push_back(rule1);
        
    }
    
    
    axiom = ax;
    sentence = ax;
}
void TreeGenerator::generate() {
    l *= 0.75f;
    std::string nextSentence = "";
    for (int i = 0; i < sentence.length(); i++) {
        char current = sentence[i];
        bool found = false;
        for (int j = 0; j < rules.size(); j++) {
            if (current == rules[j].a) {
                found = true;
                nextSentence += rules[j].b;
                break;
            }
        }
        if (!found) {
            nextSentence += current;
        }
    }
    sentence = nextSentence;
    turtle();
}

void TreeGenerator::turtle() {
    //pts.push_back(pt);

    for (int i = 0; i < sentence.length(); i++) {
        char current = sentence[i];
        if (current == 'F') {

            pt += glm::vec3(l * sin(theta * rot), l * cos(theta * rot), 0.0f);
            pts.push_back(pt);
        }
        else if (current == '+') {
            rot--;
        }
        else if (current == '-') {
            rot++;
        }
        else if (current == '[') {
            s.push(glm::vec4(pt, rot));

        }
        else if (current == ']') {
            if (!s.empty()) {
                glm::vec4 temp = s.top();
                pt = glm::vec3(temp.x, temp.y, temp.z);
                s.pop();
                pts.push_back(pt);
                rot = temp.w;
            }



        }
    }
}

void TreeGenerator::iterate(int iterations) {
    turtle();
    for (int i = 0; i < iterations; i++) {
        generate();
    }
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * pts.size(), pts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glBindVertexArray(0);
}

void TreeGenerator::draw() {
    glBindVertexArray(vao);
    glLineWidth(3.0f);
    glDrawArrays(GL_LINE_LOOP, 0, pts.size());
    glBindVertexArray(0);
}