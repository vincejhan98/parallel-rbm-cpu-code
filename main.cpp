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

void init_weights_bias(int num_nodes, std::vector<std::vector<float>>& weights,
               std::vector<std::vector<float>>& weights_T,
               float visible_bias[], float hidden_bias[]) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0, 100.0); // Distribution for float values between -100 and 100

    // Initialize weights matrix and its transpose
    weights.resize(num_nodes, std::vector<float>(num_nodes));
    weights_T.resize(num_nodes, std::vector<float>(num_nodes));
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            weights[i][j] = dis(gen); // Random float value for weight
            weights_T[j][i] = weights[i][j]; // Transpose
        }
    }

    // Initialize visible bias
    for (int i = 0; i < num_nodes; ++i) {
        visible_bias[i] = dis(gen); // Random float value for visible bias
    }

    // Initialize hidden bias
    for (int i = 0; i < num_nodes; ++i) {
        hidden_bias[i] = dis(gen); // Random float value for hidden bias
    }
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
    int seed = find_int_arg(argc, argv, "-s", 7);
    int iterations = find_int_arg(argc, argv, "-i", 50000);

    // Initial nodes setup
    int num_nodes = find_int_arg(argc, argv, "-n", 150);
    float* visibles = new float[num_nodes];
    float* hiddens = new float[num_nodes];
    init_nodes(num_nodes, seed, visibles);
    init_nodes(num_nodes, seed, hiddens);

    // Experiment 2
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> weights_T;
    float visible_bias[num_nodes];
    float hidden_bias[num_nodes];
    // randomize weighs and bias
    init_weights_bias(num_nodes, weights, weights_T, visible_bias, hidden_bias);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    #ifdef _OPENMP
    #pragma omp parallel default(shared)
    #endif
    {
        for (int step = 0; step < iterations; ++step) {
            simulate_one_step(num_nodes, visibles, hiddens, weights, weights_T, visible_bias, hidden_bias);
            
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
    std::cout << "Simulation Time = " << seconds << " seconds for " << num_nodes << " nodes and " << iterations << " iterations\n";
    fsave.close();
}