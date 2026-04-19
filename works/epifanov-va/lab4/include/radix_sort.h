#pragma once

#define BLOCK_SIZE 256

template <typename T>
T* radix_sort(T* d_input, T* d_output, int n);
