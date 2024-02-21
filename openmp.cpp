#include "common.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Grid Square Data Structure
typedef struct grid_square {
    std::vector<particle_t*> parts;
    std::vector<grid_square*> neighbors;
} grid_square;

grid_square** grid;
static int grid_len;
static double grid_square_size;

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
    grid_len = static_cast<int>(size / (cutoff*8)) + 1; 
    grid_square_size = size / static_cast<double>(grid_len);

    // Initialize Grid
    grid = new grid_square*[grid_len];
    for (int i = 0; i < grid_len; i++) {
        grid[i] = new grid_square[grid_len];

        for (int j = 0; j < grid_len; j++) {
            // initializing vectors made a slight improvement
            grid[i][j].parts = {};
            grid[i][j].neighbors = {};
        }
    }

    // Initialize neighbor list per grid square
    int square_reach = static_cast<int>(cutoff  / grid_square_size) + 1;
    for (int i = 0; i < grid_len; i++) {
        for (int j = 0; j < grid_len; j++) {
            int lower_x = fmax(0, i - square_reach);
            int upper_x = fmin(grid_len-1, i + square_reach);
            int lower_y = fmax(0, j - square_reach);
            int upper_y = fmin(grid_len-1, j + square_reach);

            for (int k = lower_x; k <= upper_x; k++) {
                for (int l = lower_y; l <= upper_y; l++) {
                    grid[i][j].neighbors.push_back(&grid[k][l]);
                }
            }
        }
    }

    // assign particles to bins
    for (int i = 0; i < num_parts; i++) {
        int grid_index_x = static_cast<int>(parts[i].x/grid_square_size);
        int grid_index_y = static_cast<int>(parts[i].y/grid_square_size);

        grid[grid_index_x][grid_index_y].parts.push_back(&parts[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // 1. Apply force
    // calculate and assign threads per row and col of grid
    int total_threads = omp_get_num_threads();
    int num_threads_side = static_cast<int>( sqrt(total_threads) ); // assume num_threads is power of 2
    int grids_per_thread_side = (grid_len / num_threads_side) + 1;

    // assign row and col range for this thread
    int tid = omp_get_thread_num();
    int row_low  = (tid / num_threads_side) * grids_per_thread_side;
    int row_high = fmin(grid_len, row_low + grids_per_thread_side);
    int col_low  = (tid % num_threads_side) * grids_per_thread_side;
    int col_high  = fmin(grid_len, col_low + grids_per_thread_side);

    for (int i = row_low; i < row_high; i++) {
        for (int j = col_low; j < col_high; j++) {
            // Per particle in grid square...
            for (int k = 0; k < grid[i][j].parts.size(); k++) {
                // Zero old acceleration
                grid[i][j].parts[k]->ax = 0;
                grid[i][j].parts[k]->ay = 0;

                // Per neighbor of grid square...
                for (int l = 0; l < grid[i][j].neighbors.size(); l++) {

                    // Per neighboring grid square particle... 
                    for (int m = 0; m < grid[i][j].neighbors[l]->parts.size(); m++) {
                        apply_force(*grid[i][j].parts[k], *grid[i][j].neighbors[l]->parts[m]);
                    }
                }
            }
        }
    }
    #pragma omp barrier

    // 2. Move particles
    int parts_block_size = (num_parts / total_threads) + 1;
    int parts_min = tid * parts_block_size;
    int parts_max = fmin(num_parts, parts_min + parts_block_size);

    for (int i = parts_min; i < parts_max; ++i) {
        int old_grid_index_x = static_cast<int>(parts[i].x/grid_square_size);
        int old_grid_index_y = static_cast<int>(parts[i].y/grid_square_size);

        move(parts[i], size);

        int new_grid_index_x = static_cast<int>(parts[i].x/grid_square_size);
        int new_grid_index_y = static_cast<int>(parts[i].y/grid_square_size);

        // if grid index changed, pop it from the old grid and add to new grid
        if (old_grid_index_x != new_grid_index_x || old_grid_index_y != new_grid_index_y) {
            #pragma omp critical
            {
                // get iterator index of the element in old grid
                auto it = std::find(grid[old_grid_index_x][old_grid_index_y].parts.begin(), 
                                    grid[old_grid_index_x][old_grid_index_y].parts.end(), 
                                    &parts[i]);

                // Remove from old grid
                grid[old_grid_index_x][old_grid_index_y].parts.erase(it);

                // add particle to new grid
                grid[new_grid_index_x][new_grid_index_y].parts.push_back(&parts[i]);
            }
        }
    }
    #pragma omp barrier
}