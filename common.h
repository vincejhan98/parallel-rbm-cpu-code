#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__
#include <vector>

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

// Particle Data Structure
typedef struct particle_t {
    double x;  // Position X
    double y;  // Position Y
    double vx; // Velocity X
    double vy; // Velocity Y
    double ax; // Acceleration X
    double ay; // Acceleration Y
} particle_t;

// Simulation routine
void init_nodes(int num_nodes, int seed, float* nodes);
void simulate_one_step(int num_nodes, float* visibles, float* hiddens, std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& weights_T, float* visible_bias, float* hidden_bias);
#endif
