#pragma once

#include <stdio.h>
#include <cmath>
#include <vector>
#include <GL/glew.h>

namespace CPUBoids {
    void initSimulation(int N);
    void stepSimulationScatteredGrid(float dt, float r1, float r2, float r3);
    void copyBoidsToVBO(GLuint vbodptr_positions, GLuint vbodptr_velocities);

    void endSimulation();
}
