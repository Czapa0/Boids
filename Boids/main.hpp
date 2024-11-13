#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility/glslUtility.hpp"
#include "kernel.h"
#include "CPUkernel.h"
#include <Windows.h>
#include "ocornut-imgui-b81bd7e/imgui.h"
#include "ocornut-imgui-b81bd7e/imgui_impl_glfw.h"
#include "ocornut-imgui-b81bd7e/imgui_impl_opengl3.h"
#include <string>

// GL vars

GLuint positionLocation = 0;
GLuint velocitiesLocation = 1;
const char *attributeLocations[] = { "Position", "Velocity" };

GLuint boidVAO = 0;
GLuint boidVBO_positions = 0;
GLuint boidVBO_velocities = 0;
GLuint boidIBO = 0;
GLuint displayImage;
GLuint program[2];

const unsigned int PROG_BOID = 0;

const float fovy = (float) (PI / 4);
const float zNear = 0.10f;
const float zFar = 10.0f;
int width = 1280;
int height = 750;
int pointSize = 2;

// Camera controls
bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;
float theta = 1.22f;
float phi = -0.70f;
float zoom = 4.0f;
glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraPosition;

glm::mat4 projection;

const char *projectName;

// main
int main(int argc, char* argv[]);

// Animation step
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void updateCamera();
void runCUDA(float r1, float r2, float r3);
void runCPU(float r1, float r2, float r3);

// Init
bool init(int argc, char **argv);
void initVAO();
void initShaders(GLuint *program);
bool LoadConfigs();