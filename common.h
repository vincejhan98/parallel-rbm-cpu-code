#include <vector>
#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__
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
// CUSTOM
#define grid_square_size (cutoff)
#define grid_len (static_cast<int>(size / grid_square_size) + 1)
//Grid Square Data Structure
typedef struct grid_square {
    std::vector<particle_t> parts;
    std::vector<grid_square> neighbors;
    // int num_parts;
} grid_square;
// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size);
void simulate_one_step(particle_t* parts, int num_parts, double size);
#endif