#include "common.h"
#include <omp.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <cblas.h>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void init_nodes(int num_nodes, int seed, float* nodes) {
    std::default_random_engine rng(seed); // Initialize Mersenne Twister random number generator with seed
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f); // Uniform distribution for float values between 0 and 1

    // Generate random float values for each node
    for (int i = 0; i < num_nodes; i++) {
        nodes[i] = std::round(distribution(rng)); // Generate a random float value between 0 and 1
    }
}

void simulate_one_step(int num_nodes, float* visibles, float* hiddens, std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& weights_T, float* visible_bias, float* hidden_bias, float clamp) {
    // assign nodes to threads
    int tid = omp_get_thread_num();
    if (tid >= num_nodes) return;
    int total_threads = omp_get_num_threads();
    int nodes_per_thread = (num_nodes / total_threads) + 1;

    int node_low  = tid * nodes_per_thread;
    int node_high = std::min(num_nodes, node_low + nodes_per_thread);
    int nodes_assigned_cnt = node_high - node_low;

    // copy visible nodes temporarilly
    float incoming_visibles[num_nodes];
    std::copy(visibles, visibles + num_nodes, incoming_visibles);

    // for random sampling 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    // =================================
    // 1. Parallelized Visible to Hidden
    // =================================
    float random_val;
    for (int i = node_low; i < node_high; i++) {
        // forward matrix * vector and add
        double fwd_mva = cblas_sdot(num_nodes, weights[i].data(), 1, visibles, 1);
        
        // add bias
        fwd_mva += hidden_bias[i];

        // probability through sigmoid
        float fwd_prob = sigmoid(fwd_mva);

        // sampling through comparison with random values
        random_val = distribution(gen);
        hiddens[i] = (fwd_prob > random_val) ? 1 : 0;
    }
    #pragma omp barrier

    // =================================
    // 2. Parallelized Visible to Hidden
    // =================================
    for (int i = node_low; i < node_high; i++) {
        // reverse matrix * vector and add
        double rev_mva = cblas_sdot(num_nodes, hiddens, 1, weights_T[i].data(), 1);

        // add bias
        rev_mva += visible_bias[i];

        // probability through sigmoid
        float rev_prob = sigmoid(rev_mva);

        // sampling through comparison with random values
        random_val = distribution(gen);
        visibles[i] = (rev_prob > random_val) ? 1 : 0;
    }
    #pragma omp barrier

    // clamp
    // visibles[2] = clamp;
}