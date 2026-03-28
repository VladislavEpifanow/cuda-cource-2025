#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "radix_sort.h"

#define CUDA_CHECK(err) \
if(err != cudaSuccess){ \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(-1); \
}

template <typename T>
void test_radix_sort(size_t N) {
    printf("Testing %zu elements of type %s\n", N, (sizeof(T) == 4 ? "int32" : "int64"));

    T* h_data = new T[N];
    T* h_out = new T[N];
    T* h_ref = new T[N];

    // fill with random data
    for (size_t i = 0; i < N; i++) {
        h_data[i] = rand() % 100000 - 50000;
        h_ref[i] = h_data[i];
    }

    // CPU sort
    clock_t start = clock();
    qsort(h_ref, N, sizeof(T), [](const void* a, const void* b) {
        T va = *(T*)a, vb = *(T*)b;
        return (va > vb) - (va < vb);
    });
    double cpu_time = (double)(clock() - start)/CLOCKS_PER_SEC;
    printf("CPU qsort time: %.6f sec\n", cpu_time);

    // GPU memory
    T* d_input; 
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, N*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, N*sizeof(T), cudaMemcpyHostToDevice));

    // GPU radix sort
    cudaEvent_t gstart, gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);

    cudaEventRecord(gstart);
    radix_sort<T>(d_input, d_output, N);
    cudaEventRecord(gstop);
    cudaEventSynchronize(gstop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gstart, gstop);
    gpu_time /= 1000.0f; // convert to seconds

    CUDA_CHECK(cudaMemcpy(h_out, d_input, N*sizeof(T), cudaMemcpyDeviceToHost));
    
    // check correctness
    bool correct = true;
    for (size_t i = 1; i < N; i++) {
        if (h_out[i-1] > h_out[i]) { correct = false; break; }
    }

    printf("GPU Radix Sort time: %.6f sec, Correctness: %s\n", gpu_time, correct ? "PASS" : "FAIL");
    printf("Speedup (CPU / Custom GPU): %.2fx\n", cpu_time / gpu_time);

    // Thrust sort
    float gpu_thrust_time;
    CUDA_CHECK(cudaMemcpy(d_input, h_data, N*sizeof(T), cudaMemcpyHostToDevice));
    cudaEventRecord(gstart);
    thrust::sort(thrust::device, d_input, d_input+N);
    cudaEventRecord(gstop);
    cudaEventSynchronize(gstop);
    cudaEventElapsedTime(&gpu_thrust_time, gstart, gstop);
    gpu_thrust_time /= 1000.0f;

    CUDA_CHECK(cudaMemcpy(h_out, d_input, N*sizeof(T), cudaMemcpyDeviceToHost));

    // check correctness
    correct = true;
    for (size_t i = 1; i < N; i++) {
        if (h_out[i-1] > h_out[i]) { correct = false; break; }
    }

    printf("Thrust sort time: %.6f sec, Correctness: %s\n", gpu_thrust_time, correct ? "PASS" : "FAIL");
    printf("Speedup (CPU / Thrust): %.2fx\n", cpu_time / gpu_thrust_time);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_data;
    delete[] h_out;
    delete[] h_ref;

    printf("\n");
}

int main() {
    // test sizes: 1e5, 5e5, 1e6 as per requirements
    size_t sizes[] = {100000, 500000, 1000000};

    for (size_t i = 0; i < 3; i++) {
        size_t N = sizes[i];
        test_radix_sort<int32_t>(N);
        test_radix_sort<int64_t>(N);
    }

    return 0;
}