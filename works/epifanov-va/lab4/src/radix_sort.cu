#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 512

#define CUDA_CHECK(err) \
if(err != cudaSuccess){ \
    printf("CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(-1); \
}

// Extract bit
template <typename T>
__device__ inline int get_bit(T val, int bit) {
    // standard bit extraction
    return (val >> bit) & 1;
}

// Mark elements: 1 if bit = 0
template <typename T>
__global__ void mark_bit(T* input, int* is_zero, int bit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = get_bit(input[idx], bit);

        // fix for negative numbers (sign bit)
        if (bit == sizeof(T)*8 - 1) b ^= 1;

        is_zero[idx] = 1 - b;
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

// multi_block_scan - recursive scan for large arrays
void multi_block_scan(int* d_in, int* d_out, int n) {
    int numBlocks = (n + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE);

    int* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(int)));

    // Scan inside blocks
    block_scan<<<numBlocks, BLOCK_SIZE, 2*BLOCK_SIZE*sizeof(int)>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (numBlocks > 1) {
        int* d_block_sums_scanned;
        CUDA_CHECK(cudaMalloc(&d_block_sums_scanned, numBlocks * sizeof(int)));

        // recursive sums arrays scan
        multi_block_scan(d_block_sums, d_block_sums_scanned, numBlocks);

        // Add corrected sums to all blocks
        add_block_sums<<<numBlocks, BLOCK_SIZE>>>(d_out, d_block_sums_scanned, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_block_sums_scanned);
    }

    cudaFree(d_block_sums);
}

// Scatter: compute final positions and reorder elements
template <typename T>
__global__ void scatter(T* input, T* output,
                       int* scan_zero, int* is_zero,
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
void radix_sort(T* d_input, T* d_output, int n) {
    // Prepare data
    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;

    int* d_is_zero;
    int* d_scan_zero;
    
    CUDA_CHECK(cudaMalloc(&d_is_zero, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan_zero, n * sizeof(int)));

    int num_bits = sizeof(T) * 8;
    // go fo every bit of integer
    for (int bit = 0; bit < num_bits; bit++) {

        mark_bit<T><<<blocks, threads>>>(d_input, d_is_zero, bit, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // scan for is_zero array
        multi_block_scan(d_is_zero, d_scan_zero, n);

        int total_zero;
        int last;
        // get total zeros in input current bits and last element
        CUDA_CHECK(cudaMemcpy(&total_zero, &d_scan_zero[n-1], sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, &d_is_zero[n-1], sizeof(int), cudaMemcpyDeviceToHost));

        total_zero += last;

        scatter<T><<<blocks, threads>>>(
            d_input, d_output,
            d_scan_zero, d_is_zero,
            total_zero, n
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // update source array
        T* tmp = d_input;
        d_input = d_output;
        d_output = tmp;
    }

    cudaFree(d_is_zero);
    cudaFree(d_scan_zero);
}

// Explicit template instantiation
template void radix_sort<int>(int* d_input, int* d_output, int n);
template void radix_sort<int64_t>(int64_t* d_input, int64_t* d_output, int n);