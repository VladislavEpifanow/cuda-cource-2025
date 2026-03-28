#pragma once

#define BLOCK_SIZE 256
#define BITS_PER_PASS 4
#define RADIX (1 << BITS_PER_PASS)

template <typename T>
void radix_sort(T* d_input, T* d_output, int n);