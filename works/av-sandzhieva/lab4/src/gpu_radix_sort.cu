#include "../include/gpu_radix_sort.h"
#include "../include/utils.h"

template<typename T>
__global__ void flip_msb(T* data, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if constexpr (sizeof(T) == 4)
        reinterpret_cast<uint32_t*>(data)[i] ^= 0x80000000u;
    else
        reinterpret_cast<uint64_t*>(data)[i] ^= 0x8000000000000000ull;
}

template<typename T>
__device__ __forceinline__ unsigned int get_digit(T x, unsigned int shift) {
    if constexpr (sizeof(T) == 4)
        return (reinterpret_cast<const uint32_t&>(x) >> shift) & 0xFFu;
    else
        return (reinterpret_cast<const uint64_t&>(x) >> shift) & 0xFFu;
}

template<typename T>
__global__ void compute_histogram(const T* in, unsigned int* digitsPerBlock,
                                  size_t n, unsigned int shift, unsigned int BLOCKS_COUNT) {
    __shared__ unsigned int s_warp_hist[WARPS_PER_BLOCK][RADIX];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    for (int i = tid; i < WARPS_PER_BLOCK * RADIX; i += BLOCK_SIZE)
        ((unsigned int*)s_warp_hist)[i] = 0;
    __syncthreads();

    size_t col = blockIdx.x * BLOCK_SIZE + tid;
    unsigned int digit = 0xFFFFFFFFu;
    if (col < n) digit = get_digit(in[col], shift);

    unsigned int peers_bits = 0;
    #pragma unroll
    for (int i = 0; i < WARP_SIZE; i++) {
        unsigned int other = __shfl_sync(0xFFFFFFFF, digit, i);
        if (other == digit) peers_bits |= (1u << i);
    }

    unsigned int count_in_warp = __popc(peers_bits);
    int first_lane = __ffs(peers_bits) - 1;

    if (lane == first_lane && col < n) {
        s_warp_hist[warp_id][digit] = count_in_warp;
    }
    __syncthreads();

    if (tid < RADIX) {
        unsigned int total = 0;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++)
            total += s_warp_hist[w][tid];

        digitsPerBlock[tid * BLOCKS_COUNT + blockIdx.x] = total;
    }
}

__global__ void scan_histogram_blelloch(unsigned int* digitsPerBlock,
                                        unsigned int* totalCountPerDigit,
                                        unsigned int BLOCKS_COUNT) {
    extern __shared__ unsigned int temp[];

    int tid = threadIdx.x;
    int digit = blockIdx.x;
    unsigned int offset_idx = digit * BLOCKS_COUNT;

    int ai = tid;
    int bi = tid + blockDim.x;

    temp[ai] = (ai < BLOCKS_COUNT) ? digitsPerBlock[offset_idx + ai] : 0;
    temp[bi] = (bi < BLOCKS_COUNT) ? digitsPerBlock[offset_idx + bi] : 0;

    int n = 2 * blockDim.x;
    int offset = 1;

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            temp[bi2] += temp[ai2];
        }
        offset *= 2;
    }

    __syncthreads();
    if (tid == 0) {
        totalCountPerDigit[digit] = temp[n - 1];
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }
    __syncthreads();

    if (ai < BLOCKS_COUNT) digitsPerBlock[offset_idx + ai] = temp[ai];
    if (bi < BLOCKS_COUNT) digitsPerBlock[offset_idx + bi] = temp[bi];
}

__global__ void scan_buckets(unsigned int* totalCountPerDigit) {
    __shared__ unsigned int temp[RADIX];
    int tid = threadIdx.x;
    int n = RADIX;
    int ai = tid;
    int bi = tid + (n / 2);

    temp[ai] = totalCountPerDigit[ai];
    temp[bi] = totalCountPerDigit[bi];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            temp[bi2] += temp[ai2];
        }
        offset *= 2;
    }
    if (tid == 0) temp[n - 1] = 0;

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }
    __syncthreads();

    totalCountPerDigit[ai] = temp[ai];
    totalCountPerDigit[bi] = temp[bi];
}

template<typename T>
__global__ void scatter(const T* in, T* out,
                        const unsigned int* digitsPerBlock,
                        const unsigned int* bucketOffsets,
                        size_t n, unsigned int shift, unsigned int BLOCKS_COUNT) {
    __shared__ unsigned int s_warp_counts[WARPS_PER_BLOCK][RADIX];
    __shared__ unsigned int s_global_bases[RADIX];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    for (int i = tid; i < WARPS_PER_BLOCK * RADIX; i += BLOCK_SIZE)
        ((unsigned int*)s_warp_counts)[i] = 0;

    if (tid < RADIX) {
        s_global_bases[tid] = bucketOffsets[tid] + digitsPerBlock[tid * BLOCKS_COUNT + blockIdx.x];
    }
    __syncthreads();

    size_t col = blockIdx.x * BLOCK_SIZE + tid;
    bool active = col < n;
    T val = 0;
    unsigned int digit = 0;

    unsigned int rank_in_warp = 0;
    unsigned int count_in_warp = 0;

    if (active) {
        val = in[col];
        digit = get_digit(val, shift);

        unsigned int peers_bits = 0;
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; i++) {
            unsigned int other = __shfl_sync(0xFFFFFFFF, digit, i);
            if (other == digit) peers_bits |= (1u << i);
        }

        rank_in_warp = __popc(peers_bits & ((1u << lane) - 1));
        count_in_warp = __popc(peers_bits);

        if (rank_in_warp == count_in_warp - 1) {
            s_warp_counts[warp_id][digit] = count_in_warp;
        }
    }
    __syncthreads();

    if (active) {
        unsigned int rank_prev_warps = 0;
        #pragma unroll
        for (int w = 0; w < warp_id; w++)
            rank_prev_warps += s_warp_counts[w][digit];

        unsigned int final_pos = s_global_bases[digit] + rank_prev_warps + rank_in_warp;
        out[final_pos] = val;
    }
}


template<typename T>
void gpu_radix_sort(T* d_out, T* d_in, T* d_temp,
                    unsigned int* d_digitsPerBlock,
                    unsigned int* d_totalCountPerDigit,
                    size_t n) 
{
    if (n == 0) return;

    const unsigned int passes = (sizeof(T) * 8) / BITS_PER_PASS;
    const unsigned int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (num_blocks > 2048) {
        printf("Error: n too large for single-block scan\n");
        return; 
    }

    CUDA_CHECK(cudaMemcpy(d_temp, d_in, n * sizeof(T), cudaMemcpyDeviceToDevice));

    flip_msb<<<num_blocks, BLOCK_SIZE>>>(d_temp, n);
    CUDA_CHECK(cudaGetLastError());
    
    unsigned int scan_threads = 32;
    while (scan_threads < num_blocks) scan_threads *= 2;
    scan_threads = min(scan_threads, 1024u);
    size_t scan_smem = 2 * scan_threads * sizeof(unsigned int);

    T *src = d_temp, *dst = d_out;

    for (unsigned int p = 0; p < passes; ++p) {
        unsigned int shift = p * BITS_PER_PASS;

        compute_histogram<<<num_blocks, BLOCK_SIZE>>>(src, d_digitsPerBlock, n, shift, num_blocks);
        CUDA_CHECK(cudaGetLastError());

        scan_histogram_blelloch<<<RADIX, scan_threads, scan_smem>>>(
            d_digitsPerBlock, d_totalCountPerDigit, num_blocks);
        CUDA_CHECK(cudaGetLastError());

        scan_buckets<<<1, RADIX / 2>>>(d_totalCountPerDigit);
        CUDA_CHECK(cudaGetLastError());

        scatter<<<num_blocks, BLOCK_SIZE>>>(src, dst, d_digitsPerBlock, d_totalCountPerDigit, n, shift, num_blocks);
        CUDA_CHECK(cudaGetLastError());

        T* tmp = src; src = dst; dst = tmp;
    }

    if (src != d_out) {
        CUDA_CHECK(cudaMemcpy(d_out, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    flip_msb<<<num_blocks, BLOCK_SIZE>>>(d_out, n);
    CUDA_CHECK(cudaGetLastError());
}

template void gpu_radix_sort<int32_t>(int32_t*, int32_t*, int32_t*, unsigned int*, unsigned int*, size_t);
template void gpu_radix_sort<int64_t>(int64_t*, int64_t*, int64_t*, unsigned int*, unsigned int*, size_t);
