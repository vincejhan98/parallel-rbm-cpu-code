#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <omp.h>
// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, float* nodes, int num_nodes) {

    for (int i = 0; i < num_nodes; ++i) {
        fsave << nodes[i] << " ";
    }

    fsave << std::endl;
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of nodes" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set visible node initialization seed" << std::endl;
        std::cout << "-i <int>: iterations" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Experiment Settings
    int seed = find_int_arg(argc, argv, "-s", 0);
    int iterations = find_int_arg(argc, argv, "-i", 1000);

    // Initial nodes setup
    int num_nodes = find_int_arg(argc, argv, "-n", 3);
    float* visibles = new float[num_nodes];
    float* hiddens = new float[num_nodes];
    init_nodes(num_nodes, seed, visibles);
    init_nodes(num_nodes, seed, hiddens);

    // Experiment 1
    std::vector<std::vector<float>> weights = {{-9, -9, -1},
                                              {-12, 4, -10},
                                              {4, -12, -10}};
    std::vector<std::vector<float>> weights_T = {{-9, -12, 4},
                                               {-9, 4, -12},
                                               {-1, -10, -10}};
    float visible_bias[] = {6, 6, 4};
    float hidden_bias[] = {4, 6, 6};
    float clamp = 1;

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    #ifdef _OPENMP
    #pragma omp parallel default(shared)
    #endif
    {
        for (int step = 0; step < iterations; ++step) {
            simulate_one_step(num_nodes, visibles, hiddens, weights, weights_T, visible_bias, hidden_bias, clamp);
            
            // Save state if necessary
    #ifdef _OPENMP
    #pragma omp master
    #endif
            if (fsave.good()) {
                save(fsave, visibles, num_nodes);
            }
            
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " << num_nodes << " particles.\n";
    fsave.close();
}