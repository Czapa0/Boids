#include <stdio.h>
#include <random>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "utilityCore.hpp"
#include "CPUkernel.h"
#include <algorithm>
#include <iostream>

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

namespace CPUBoids {
    // Some globals

    // Block size for kernels
#define blockSize 128

#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define maxSpeed 1.0f

// Scene size for calculations
#define scene_scale 100.0f

    int numObjects;
    dim3 threadsPerBlock(blockSize);

    glm::vec3* position; // position buffer
    glm::vec3* velocity_1; // 1. velocity buffer (for reading)
    glm::vec3* velocity_2; // 2. velocity buffer (for writing)

    int* boidArrayIndices; // What index in position and dev_velX represents this particle?
    int* boidGridIndices; // What grid cell is this particle in?

    int* gridCellStartIndices; // Start of cell in boidArrayIndices
    int* gridCellEndIndices;   // End of cell in boidArrayIndices

    int gridCellCount;
    int gridSideCount;
    float gridCellWidth;
    float halfGridCellWidth;
    float gridInverseCellWidth;
    glm::vec3 gridMinimum;

    // useful for random data generation
    unsigned int hash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }

    glm::vec3 generateRandomVec3(float time, int index) {
        unsigned int seed = hash(index * static_cast<int>(time)) + static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::default_random_engine rng(seed);
        std::uniform_real_distribution<float> unitDistrib(-1.0f, 1.0f);

        return glm::vec3(unitDistrib(rng), unitDistrib(rng), unitDistrib(rng));
    }

    // random position
    void GenerateRandomPositionArray(int time, int index, glm::vec3* arr, float scale) {
        glm::vec3 rand = generateRandomVec3(time, index);
        arr[index].x = scale * rand.x;
        arr[index].y = scale * rand.y;
        arr[index].z = scale * rand.z;
    }

    // init
    void CPUBoids::initSimulation(int N) {
        numObjects = N;
        dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

        position = static_cast<glm::vec3*>(malloc(N * sizeof(glm::vec3)));

        velocity_1 = static_cast<glm::vec3*>(malloc(N * sizeof(glm::vec3)));

        velocity_2 = static_cast<glm::vec3*>(malloc(N * sizeof(glm::vec3)));

        for (int i = 0; i < N; ++i) {
            GenerateRandomPositionArray(1, i, position, scene_scale);
        }

        halfGridCellWidth = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
        gridCellWidth = 2.0f * halfGridCellWidth;
        int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
        gridSideCount = 2 * halfSideCount;

        gridCellCount = gridSideCount * gridSideCount * gridSideCount;
        gridInverseCellWidth = 1.0f / gridCellWidth;
        float halfGridWidth = gridCellWidth * halfSideCount;
        gridMinimum.x -= halfGridWidth;
        gridMinimum.y -= halfGridWidth;
        gridMinimum.z -= halfGridWidth;

        boidArrayIndices = static_cast<int*>(malloc(N * sizeof(int)));

        boidGridIndices = static_cast<int*>(malloc(N * sizeof(int)));

        gridCellStartIndices = static_cast<int*>(malloc(gridCellCount * sizeof(int)));

        gridCellEndIndices = static_cast<int*>(malloc(gridCellCount * sizeof(int)));
    }

    // copy position buffer for drawing
    void CopyPositionsToVBO(int index, glm::vec3* pos, std::shared_ptr<GLfloat[]> vbo, float s_scale) {
        float c_scale = -1.0f / s_scale;

        vbo[4 * index + 0] = pos[index].x * c_scale;
        vbo[4 * index + 1] = pos[index].y * c_scale;
        vbo[4 * index + 2] = pos[index].z * c_scale;
        vbo[4 * index + 3] = 1.0f;
    }

    // copy velocity buffer for drawing
    void CopyVelocitiesToVBO(int index, glm::vec3* vel, std::shared_ptr<GLfloat[]> vbo, float s_scale) {
        vbo[4 * index + 0] = vel[index].x + 0.3f;
        vbo[4 * index + 1] = vel[index].y + 0.3f;
        vbo[4 * index + 2] = vel[index].z + 0.3f;
        vbo[4 * index + 3] = 1.0f;
    }

    // invoced in main.cpp
    /*void CPUBoids::copyBoidsToVBO(float* vbodptr_positions, float* vbodptr_velocities) {
        dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

        for (int i = 0; i < numObjects; ++i) {
            CopyPositionsToVBO(i, position, vbodptr_positions, scene_scale);
            CopyVelocitiesToVBO(i, velocity_1, vbodptr_velocities, scene_scale);
        }
    }*/
    void CPUBoids::copyBoidsToVBO(GLuint vbodptr_positions, GLuint vbodptr_velocities) {
        dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
        //glNamedBufferData(vbodptr_positions, 4 * numObjects * sizeof(GLfloat), position, GL_DYNAMIC_DRAW);
        //glNamedBufferData(vbodptr_velocities, 4 * numObjects * sizeof(GLfloat), velocity_1, GL_DYNAMIC_DRAW);
        std::shared_ptr<GLfloat[]> pos{ new GLfloat[4 * numObjects] };
        std::shared_ptr<GLfloat[]> vel{ new GLfloat[4 * numObjects] };
        for (int i = 0; i < numObjects; ++i) {
            CopyPositionsToVBO(i, position, pos, scene_scale);
            CopyVelocitiesToVBO(i, velocity_1, vel, scene_scale);
        }
        glNamedBufferData(vbodptr_positions, 4 * numObjects * sizeof(GLfloat), pos.get(), GL_DYNAMIC_DRAW);
        glNamedBufferData(vbodptr_velocities, 4 * numObjects * sizeof(GLfloat), vel.get(), GL_DYNAMIC_DRAW);
    }

    // position update
    void UpdatePosition(int index, float dt, glm::vec3* pos, glm::vec3* vel) {
        glm::vec3 thisPos = pos[index];
        glm::vec3 thisVel = vel[index];
        thisPos += thisVel * dt;

        // edge event
        if (thisPos.x < -scene_scale ||
            thisPos.y < -scene_scale ||
            thisPos.z < -scene_scale ||
            thisPos.x > scene_scale ||
            thisPos.y > scene_scale ||
            thisPos.z > scene_scale) {
            vel[index] = -thisVel;
        }
        else {
            pos[index] = thisPos;
        }
    }

    // grid unic 1D index from 3D index
    int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
        return x + y * gridResolution + z * gridResolution * gridResolution;
    }

    void ComputeIndices(int index, int gridResolution,
        glm::vec3 gridMin, float inverseCellWidth,
        glm::vec3* pos, int* indices, int* gridIndices) {
        glm::vec3 p = pos[index];
        // 3D indices for grid cell
        int x = imin(glm::floor((p.x - gridMin.x) * inverseCellWidth), gridResolution - 1);
        int y = imin(glm::floor((p.y - gridMin.y) * inverseCellWidth), gridResolution - 1);
        int z = imin(glm::floor((p.z - gridMin.z) * inverseCellWidth), gridResolution - 1);

        gridIndices[index] = gridIndex3Dto1D(x, y, z, gridResolution);
        indices[index] = index;
    }

    // reset buffer
    void ResetIntBuffer(int index, int* intBuffer, int value) {
        intBuffer[index] = value;
    }

    void IdentifyCellStartEnd(int index, int* particleGridIndices,
        int* gridCellStartIndices, int* gridCellEndIndices) {
        int cell1 = particleGridIndices[index];
        int cell2 = particleGridIndices[index + 1];

        // cell change
        if (cell1 != cell2) {
            gridCellEndIndices[cell1] = index;
            gridCellStartIndices[cell2] = index + 1;
        }
    }

    void UpdateVelNeighborSearchScattered(
        int index, int gridResolution, glm::vec3 gridMin,
        float inverseCellWidth, float cellWidth,
        int* gridCellStartIndices, int* gridCellEndIndices,
        int* particleArrayIndices, float r1, float r2, float r3,
        glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
        index = particleArrayIndices[index];
        glm::vec3 p = pos[index];

        // boid's grid cell identification
        int x = imin(glm::floor((p.x - gridMin.x) * inverseCellWidth), gridResolution - 1);
        int y = imin(glm::floor((p.y - gridMin.y) * inverseCellWidth), gridResolution - 1);
        int z = imin(glm::floor((p.z - gridMin.z) * inverseCellWidth), gridResolution - 1);
        int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);

        // determine cells that need checking
        float dist = cellWidth / 2;
        glm::vec3 maxGrid(0, 0, 0), minGrid(gridResolution - 1, gridResolution - 1, gridResolution - 1);
        for (int i = -1; i <= 1; i += 2) {
            for (int j = -1; j <= 1; j += 2) {
                for (int k = -1; k <= 1; k += 2) {
                    glm::vec3 v = p + glm::vec3(i * dist, j * dist, k * dist); // cube vertex
                    // grid indices
                    v.x = imax(imin(glm::floor((v.x - gridMin.x) * inverseCellWidth), gridResolution - 1), 0);
                    v.y = imax(imin(glm::floor((v.y - gridMin.y) * inverseCellWidth), gridResolution - 1), 0);
                    v.z = imax(imin(glm::floor((v.z - gridMin.z) * inverseCellWidth), gridResolution - 1), 0);
                    minGrid = glm::min(minGrid, v);
                    maxGrid = glm::max(maxGrid, v);
                }
            }
        }

        int gridIndZ = gridIndex3Dto1D(minGrid.x, minGrid.y, minGrid.z, gridResolution);

        int powGridResolution = gridResolution * gridResolution;

        glm::vec3 center(0.0, 0.0, 0.0);
        glm::vec3 v2(0.0, 0.0, 0.0);
        glm::vec3 avg_velocity(0.0, 0.0, 0.0);
        int neighbours1 = 0;
        int neighbours3 = 0;
        for (int i = minGrid.z; i <= maxGrid.z; ++i) {
            int gridIndY = gridIndZ;
            for (int j = minGrid.y; j <= maxGrid.y; ++j) {
                int gridIndX = gridIndY;
                for (int k = minGrid.x; k <= maxGrid.x; ++k) {
                    // start/end of grid cell
                    int start = gridCellStartIndices[gridIndX];
                    int end = gridCellEndIndices[gridIndX];

                    for (int l = start; l <= end; ++l) {
                        if (l == -1) {
                            break;
                        }

                        int nInd = particleArrayIndices[l];
                        //std::cout << nInd << std::endl;
                        if (index == nInd) {
                            continue;
                        }

                        glm::vec3 nPos = pos[nInd];
                        float dist = glm::distance(p, nPos);

                        // rule 1
                        if (dist <= rule1Distance)
                        {
                            center += nPos;
                            neighbours1++;
                        }
                        // rule 2
                        if (dist <= rule2Distance)
                        {
                            v2 -= (nPos - p);
                        }
                        // rule 3
                        if (dist <= rule3Distance)
                        {
                            avg_velocity += vel1[nInd];
                            neighbours3++;
                        }
                    }

                    ++gridIndX;
                }

                gridIndY += gridResolution;
            }

            gridIndZ += powGridResolution;
        }

        if (neighbours1 > 0)
        {
            center /= neighbours1;
        }
        glm::vec3 v1 = (center - p) * r1;

        v2 *= r2;
        if (neighbours3 > 0)
        {
            avg_velocity /= neighbours3;
        }

        glm::vec3 v3 = avg_velocity * r3;

        glm::vec3 vel_change = v1 + v2 + v3 + vel1[index];

        // Speed change with regard to maxSpeed
        float speed = glm::length(vel_change);
        if (speed > maxSpeed) {
            vel_change = (vel_change / speed) * maxSpeed;
        }
        vel2[index] = vel_change;
    }

    void sort_by_key(int* gridIndices, int* arrayIndices, int numObjects) {
        // Create a custom comparison function or lambda to sort by gridIndices
        auto compareFunction = [&gridIndices](int a, int b) {
            return gridIndices[a] < gridIndices[b];
            };

        // Sort indices based on the comparison function
        std::sort(arrayIndices, arrayIndices + numObjects, compareFunction);

        int* tempGridIndices = new int[numObjects];
        for (int i = 0; i < numObjects; ++i) {
            tempGridIndices[i] = gridIndices[arrayIndices[i]];
        }

        for (int i = 0; i < numObjects; ++i) {
            gridIndices[i] = tempGridIndices[i];
        }

        delete[] tempGridIndices;
    }

    void CPUBoids::stepSimulationScatteredGrid(float dt, float r1, float r2, float r3) {
        // indices for boids
        for (int i = 0; i < numObjects; ++i) {
            ComputeIndices(i, gridSideCount, gridMinimum, gridInverseCellWidth, position, boidArrayIndices, boidGridIndices);
        }

        // sort boidArrayIndices by boidGridIndices
        sort_by_key(boidGridIndices, boidArrayIndices, numObjects);

        for (int i = 0; i < gridCellCount; ++i) {
            ResetIntBuffer(i, gridCellStartIndices, -1);
            ResetIntBuffer(i, gridCellEndIndices, -1);
        }
        // start/end of cells
        gridCellStartIndices[boidGridIndices[0]] = 0;
        for (int i = 0; i < numObjects - 1; ++i) {
            IdentifyCellStartEnd(i, boidGridIndices, gridCellStartIndices, gridCellEndIndices);
        }
        gridCellEndIndices[boidGridIndices[numObjects - 1]] = numObjects - 1;

        // velocity update
        for (int i = 0; i < numObjects; ++i) {
            UpdateVelNeighborSearchScattered(i, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
                gridCellStartIndices, gridCellEndIndices, boidArrayIndices, r1, r2, r3,
                position, velocity_1, velocity_2);
        }

        // position update
        for (int i = 0; i < numObjects; ++i) {
            UpdatePosition(i, dt, position, velocity_2);
        }

        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        //float milliseconds = 0;
        //cudaEventElapsedTime(&milliseconds, start, stop);
        //std::cout << milliseconds << std::endl;
        // buffer swap
        std::swap(velocity_1, velocity_2);
    }

    // free memory
    void CPUBoids::endSimulation() {
        free(velocity_1);
        free(velocity_2);
        free(position);

        free(boidArrayIndices);
        free(boidGridIndices);
        free(gridCellStartIndices);
        free(gridCellEndIndices);
    }
}