#include "common.h"
#include <cmath>
#include <iostream>
grid_square** grid;
// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}
// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}
void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    // CUSTOM
    grid = new grid_square*[grid_len];
    for (int i = 0; i < grid_len; i++) {
        grid[i] = new grid_square[grid_len];
        // for (int j = 0; j < grid_len; j++) {
        //     grid[i][j].num_parts = 0;
        // }
    }
    // Initialize neighbor list per grid square
    int square_reach = static_cast<int>(cutoff / 2 / grid_square_size) + 1;
    for (int i = 0; i < grid_len; i++) {
        for (int j = 0; j < grid_len; j++) {
            grid_square curr_square = grid[i][j];
            int lower_x = std::max(0, i - square_reach);
            int upper_x = std::min(grid_len, i + square_reach);
            int lower_y = std::max(0, j - square_reach);
            int upper_y = std::min(grid_len, j + square_reach);
            for (int k = lower_x; k < upper_x; k++) {
                for (int l = lower_y; l < upper_y; l++) {
                    curr_square.neighbors.push_back(grid[k][l]);
                }
            }
        }
    }
    for (int i = 0; i < num_parts; i++) {
        particle_t curr_part = parts[i];
        int grid_index_x = static_cast<int>(curr_part.x/grid_square_size);
        int grid_index_y = static_cast<int>(curr_part.y/grid_square_size);
        grid_square curr_gs = grid[grid_index_x][grid_index_y];
        curr_gs.parts.push_back(curr_part);
        // curr_gs.num_parts += 1;
    }
}
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
        for (int j = 0; j < num_parts; ++j) {
            apply_force(parts[i], parts[j]);
        }
    }
    // // Move Particles
    // for (int i = 0; i < num_parts; ++i) {
    //     move(parts[i], size);
    // }
    // Compute Forces
    for (int i = 0; i < grid_len; ++i) {
        for (int j = 0; j < grid_len; ++j) {
            grid_square curr_square = grid[i][j];
            std::cout << 'hellwo';
            // Pairwise particle interactions
            for (int k = 0; k < curr_square.parts.size(); k++) {
                // std::cout << 'hellwo';
                particle_t curr_part = curr_square.parts[k];
                double old_ax = curr_part.ax;
                double old_ay = curr_part.ay;
                curr_part.ax = curr_part.ay = 0;
                for (int l = 0; l < curr_square.neighbors.size(); l++) {
                    grid_square neighbor = curr_square.neighbors[l];
                    for (int m = 0; m < neighbor.parts.size(); m++) {
                        apply_force(curr_part, neighbor.parts[m]);
                    }
                }
                // std::cout << std::abs(old_ax - curr_part.ax) << '\n';
                // if (std::abs(old_ax - curr_part.ax) > 1e-64) {
                //     std::cout << i << ' ' << j << ' ' << k;
                // }
                // if (std::abs(old_ay - curr_part.ay) > 1e-64) {
                //     std::cout << i << ' ' << j << ' ' << k;
                // }
            }
        }
    }
    // Erasing all particles within grids for reassignment (TODO: suboptimal and terrible to parallelize)
    for (int i = 0; i < grid_len; i++) {
        for (int j = 0; j < grid_len; j++) {
            grid[i][j].parts.clear();
        }
    }
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
        int grid_index_x = static_cast<int>(parts[i].x/grid_square_size);
        int grid_index_y = static_cast<int>(parts[i].y/grid_square_size);
        grid_square curr_gs = grid[grid_index_x][grid_index_y];
        curr_gs.parts.push_back(parts[i]);
    }
}