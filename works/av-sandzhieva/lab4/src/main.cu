#include "../include/utils.h"
#include "../include/gpu_radix_sort.h"

template<typename T>
void run_test(size_t n, const char* name,
              T* d_temp,                    // маллок
              unsigned int* d_digitsPerBlock,
              unsigned int* d_totalCountPerDigit) {
    printf("\n %s | %zu элементов \n", name, n);

    std::vector<T> h_in(n);
    srand(42);
    for (size_t i = 0; i < n; i++) {
        h_in[i] = (T)((rand() % 2000000000LL) - 1000000000LL);
    }


    auto h_cpu = h_in;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(h_cpu.begin(), h_cpu.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();


    T *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gpu_radix_sort(d_out, d_in, d_temp, d_digitsPerBlock, d_totalCountPerDigit, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float radix_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&radix_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    thrust::sort(thrust::device, d_in, d_in + n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float thrust_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&thrust_ms, start, stop));


    std::vector<T> h_gpu(n);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = std::equal(h_gpu.begin(), h_gpu.end(), h_cpu.begin());

    printf("CPU std::sort     : %.2f ms\n", cpu_ms);
    printf("GPU Radix Sort    : %.2f ms (%.1fx)\n", radix_ms, cpu_ms / radix_ms);
    printf("Thrust::sort      : %.2f ms\n", thrust_ms);
    printf("Correctness       : %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("Lab 4: Radix Sort on CUDA\n");

    const size_t MAX_N = 2000000;
    const unsigned int MAX_BLOCKS = (MAX_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int *d_digitsPerBlock = nullptr;
    unsigned int *d_totalCountPerDigit = nullptr;
    uint64_t *d_temp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_digitsPerBlock, RADIX * MAX_BLOCKS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_totalCountPerDigit, RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_temp, MAX_N * sizeof(uint64_t)));

    {
        size_t warm_n = 100000;
        int32_t *d_warm_in = nullptr, *d_warm_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_warm_in, warm_n * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_warm_out, warm_n * sizeof(int32_t)));

        std::vector<int32_t> h_warm(warm_n, 0); 
        CUDA_CHECK(cudaMemcpy(d_warm_in, h_warm.data(), warm_n * sizeof(int32_t), cudaMemcpyHostToDevice));

        gpu_radix_sort(d_warm_out, d_warm_in,
                       reinterpret_cast<int32_t*>(d_temp),
                       d_digitsPerBlock, d_totalCountPerDigit, warm_n);

        CUDA_CHECK(cudaFree(d_warm_in));
        CUDA_CHECK(cudaFree(d_warm_out));
    }

    size_t sizes[] = {100000, 500000, 1000000, 2000000};
    for (int i = 0; i < 4; i++) {
        run_test<int32_t>(sizes[i], "int32_t", reinterpret_cast<int32_t*>(d_temp),
                          d_digitsPerBlock, d_totalCountPerDigit);
        run_test<int64_t>(sizes[i], "int64_t", reinterpret_cast<int64_t*>(d_temp),
                          d_digitsPerBlock, d_totalCountPerDigit);
    }

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_digitsPerBlock));
    CUDA_CHECK(cudaFree(d_totalCountPerDigit));

    return 0;
}
