#include <cuda_runtime.h>
#include <stdio.h>
#include <type_traits>
#include <vector>
#include "radix_sort.h"

#define CUDA_CHECK(err) \
if(err != cudaSuccess){ \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(-1); \
}

// Extract bit
template <typename T>
__device__ inline int get_bit(T val, int bit) {
    using U = typename std::make_unsigned<T>::type;
    U uval = static_cast<U>(val);
    return static_cast<int>((uval >> bit) & U{1});
}

// Mark elements: 1 if bit = 0
template <typename T>
__global__ void mark_bit(T* input, unsigned char* is_zero, int* scan_zero, int bit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = get_bit(input[idx], bit);

        // For signed integers, invert MSB so negatives go before positives.
        if constexpr (std::is_signed_v<T>) {
            if (bit == static_cast<int>(sizeof(T) * 8 - 1)) {
                b ^= 1;
            }
        }

        int z = 1 - b;
        is_zero[idx] = static_cast<unsigned char>(z);
        scan_zero[idx] = z;
    }
}

// Blelloch scan per block (2*BLOCK_SIZE elements)
__global__ void block_scan(int* d_in, int* d_out, int* d_block_sums, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int start = 2 * blockIdx.x * blockDim.x;

    int ai = tid;
    int bi = tid + blockDim.x;

    temp[ai] = (start + ai < n) ? d_in[start + ai] : 0;
    temp[bi] = (start + bi < n) ? d_in[start + bi] : 0;

    int offset = 1;

    // upsweep
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset*(2*tid+1)-1;
            int bi2 = offset*(2*tid+2)-1;
            temp[bi2] += temp[ai2];
        }
        offset <<= 1;
    }

    // save sum for every block
    if (tid == 0) {
        if (d_block_sums)
            d_block_sums[blockIdx.x] = temp[2*blockDim.x - 1];
        temp[2*blockDim.x - 1] = 0;
    }

    // downsweep
    for (int d = 1; d <= blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset*(2*tid+1)-1;
            int bi2 = offset*(2*tid+2)-1;

            int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }
    __syncthreads();

   if (start + ai < n) d_out[start + ai] = temp[ai];
   if (start + bi < n) d_out[start + bi] = temp[bi];
}

// Add block sums
__global__ void add_block_sums(int* d_data, int* d_block_scan, int n) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int add = d_block_scan[blockIdx.x];

    if (idx < n) d_data[idx] += add;
    if (idx + blockDim.x < n) d_data[idx + blockDim.x] += add;
}

// multi_block_scan - recursive scan for large arrays.
// Uses preallocated per-level buffers to avoid cudaMalloc/cudaFree in hot path.
void multi_block_scan(int* d_data, int n, int** d_block_sums, int level) {
    int numBlocks = (n + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);

    // In-place scan: input and output are the same array.
    block_scan<<<numBlocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(int)>>>(
        d_data, d_data, d_block_sums[level], n
    );
    CUDA_CHECK(cudaGetLastError());

    if (numBlocks > 1) {
        // Recursively scan block sums in-place.
        multi_block_scan(
            d_block_sums[level], numBlocks,
            d_block_sums, level + 1
        );

        // Add scanned block sums back to each block output.
        add_block_sums<<<numBlocks, BLOCK_SIZE>>>(d_data, d_block_sums[level], n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Scatter: compute final positions and reorder elements
template <typename T>
__global__ void scatter(T* input, T* output,
                       int* scan_zero, const unsigned char* is_zero,
                       int total_zero, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int pos;

        if (is_zero[idx]) {
            pos = scan_zero[idx]; // zeros should be first
        } else {
            // place ones after zeros
            pos = total_zero + idx - scan_zero[idx];
        }

        output[pos] = input[idx];
    }
}

// Radix sort
template <typename T>
T* radix_sort(T* d_input, T* d_output, int n) {
    // Prepare data
    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;

    unsigned char* d_is_zero;
    int* d_scan_zero;
    
    CUDA_CHECK(cudaMalloc(&d_is_zero, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_scan_zero, n * sizeof(int)));

    // Build scan levels once. Each level size is the number of blocks for that scan stage.
    std::vector<int> level_blocks;
    int cur_n = n;
    do {
        int numBlocks = (cur_n + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE);
        level_blocks.push_back(numBlocks);
        cur_n = numBlocks;
    } while (cur_n > 1);

    // Allocate one workspace for all levels: d_block_sums[level].
    int total_blocks = 0;
    for (int blocks_in_level : level_blocks) {
        total_blocks += blocks_in_level;
    }

    int* d_scan_workspace;
    CUDA_CHECK(cudaMalloc(
        &d_scan_workspace,
        static_cast<size_t>(total_blocks) * sizeof(int)
    ));

    std::vector<int*> d_block_sums(level_blocks.size());

    int offset = 0;
    for (size_t level = 0; level < level_blocks.size(); ++level) {
        d_block_sums[level] = d_scan_workspace + offset;
        offset += level_blocks[level];
    }

    int num_bits = sizeof(T) * 8;
    // Go through every bit of the integer.
    for (int bit = 0; bit < num_bits; bit++) {

        mark_bit<T><<<blocks, threads>>>(d_input, d_is_zero, d_scan_zero, bit, n);
        CUDA_CHECK(cudaGetLastError());

        // scan for is_zero array
        multi_block_scan(
            d_scan_zero, n,
            d_block_sums.data(), 0
        );

        int total_zero;
        unsigned char last;
        // get total zeros in input current bits and last element
        CUDA_CHECK(cudaMemcpy(&total_zero, &d_scan_zero[n-1], sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, &d_is_zero[n-1], sizeof(unsigned char), cudaMemcpyDeviceToHost));

        total_zero += last;

        scatter<T><<<blocks, threads>>>(
            d_input, d_output,
            d_scan_zero, d_is_zero,
            total_zero, n
        );

        CUDA_CHECK(cudaGetLastError());

        // update source array
        T* tmp = d_input;
        d_input = d_output;
        d_output = tmp;
    }

    cudaFree(d_scan_workspace);
    cudaFree(d_is_zero);
    cudaFree(d_scan_zero);

    return d_input;
}

// Explicit template instantiation
template int* radix_sort<int>(int* d_input, int* d_output, int n);
template int64_t* radix_sort<int64_t>(int64_t* d_input, int64_t* d_output, int n);
