#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

// CUDA Error
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


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

glm::vec3 *position; // position buffer
glm::vec3 *velocity_1; // 1. velocity buffer (for reading)
glm::vec3 *velocity_2; // 2. velocity buffer (for writing)

int *boidArrayIndices; // What index in position and dev_velX represents this particle?
int *boidGridIndices; // What grid cell is this particle in?

int *gridCellStartIndices; // Start of cell in boidArrayIndices
int *gridCellEndIndices;   // End of cell in boidArrayIndices

int gridCellCount;
int gridSideCount;
float gridCellWidth;
float halfGridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

// useful for random data generation
__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

// random vec
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

// random position
__global__ void kernGenerateRandomPositionArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

// init
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  cudaMalloc((void**)&position, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc position failed!");

  cudaMalloc((void**)&velocity_1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc velocity_1 failed!");

  cudaMalloc((void**)&velocity_2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc velocity_2 failed!");

  kernGenerateRandomPositionArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    position, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPositionArray failed!");

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

  cudaMalloc((void**)&boidArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc boidArrayIndices failed!");

  cudaMalloc((void**)&boidGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc boidGridIndices failed!");

  cudaMalloc((void**)&gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc gridCellStartIndices failed!");

  cudaMalloc((void**)&gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc gridCellEndIndices failed!");

  cudaDeviceSynchronize();
}

// copy position buffer for drawing
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

// copy velocity buffer for drawing
__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

// invoced in main.cpp
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, position, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, velocity_1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}

// position update
__global__ void kernUpdatePosition(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

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
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    glm::vec3 p = pos[index];
    // 3D indices for grid cell
    int x = imin(glm::floor((p.x - gridMin.x) * inverseCellWidth), gridResolution - 1);
    int y = imin(glm::floor((p.y - gridMin.y) * inverseCellWidth), gridResolution - 1);
    int z = imin(glm::floor((p.z - gridMin.z) * inverseCellWidth), gridResolution - 1);

    gridIndices[index] = gridIndex3Dto1D(x, y, z, gridResolution);
    indices[index] = index;
}

// reset buffer
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N - 1) {
        return;
    }

    int cell1 = particleGridIndices[index];
    int cell2 = particleGridIndices[index + 1];

    //start of first cell
    if (index == 0) {
        gridCellStartIndices[cell1] = 0;
    }

    // cell change
    if (cell1 != cell2) {
        gridCellEndIndices[cell1] = index;
        gridCellStartIndices[cell2] = index + 1;
    }

    // end of last cell
    if (index == N - 2) {
        gridCellEndIndices[cell2] = N - 1;
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices, float r1, float r2, float r3,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N - 1) {
        return;
    }

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


void Boids::stepSimulationScatteredGrid(float dt, float r1, float r2, float r3) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    // indices for boids
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, position, boidArrayIndices, boidGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");

    // sort boidArrayIndices by boidGridIndices
    thrust::device_ptr<int> dev_thrust_grid_indices(boidGridIndices);
    thrust::device_ptr<int> dev_thrust_array_indices(boidArrayIndices);
    thrust::sort_by_key(dev_thrust_grid_indices, dev_thrust_grid_indices + numObjects, dev_thrust_array_indices);

    dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, gridCellStartIndices, -1);
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, gridCellEndIndices, -1);
    // start/end of cells
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, boidGridIndices, gridCellStartIndices, gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");
    
    // velocity update
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        gridCellStartIndices, gridCellEndIndices, boidArrayIndices, r1, r2, r3,
        position, velocity_1, velocity_2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    // position update
    kernUpdatePosition << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, position, velocity_2);
    checkCUDAErrorWithLine("kernUpdatePosition failed!");

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << milliseconds << std::endl;
    // buffer swap
    std::swap(velocity_1, velocity_2);
}

// free memory
void Boids::endSimulation() {
  cudaFree(velocity_1);
  cudaFree(velocity_2);
  cudaFree(position);

  cudaFree(boidArrayIndices);
  cudaFree(boidGridIndices);
  cudaFree(gridCellStartIndices);
  cudaFree(gridCellEndIndices);
}
