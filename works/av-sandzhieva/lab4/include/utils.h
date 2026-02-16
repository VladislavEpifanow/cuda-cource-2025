#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // UTILS_H
