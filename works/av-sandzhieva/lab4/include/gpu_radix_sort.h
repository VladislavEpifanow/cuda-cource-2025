#ifndef GPU_RADIX_SORT_H
#define GPU_RADIX_SORT_H

#include <cstddef>

#define BLOCK_SIZE         1024
#define WARP_SIZE          32
#define WARPS_PER_BLOCK    (BLOCK_SIZE / WARP_SIZE)
#define RADIX              256
#define BITS_PER_PASS      8


template<typename T>
void gpu_radix_sort(T* d_out, T* d_in, T* d_temp,
                    unsigned int* d_digitsPerBlock,
                    unsigned int* d_totalCountPerDigit,
                    size_t n);

#endif // GPU_RADIX_SORT_H
