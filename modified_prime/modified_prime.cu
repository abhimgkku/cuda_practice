#include <cuda_runtime.h>
#include <iostream>

// Kernel: test numbers for primality and store number + result in arrays
__global__ void checkPrimeKernel(long long start,
                                 int totalNumbers,
                                 long long* numbers,
                                 bool* isPrimeArr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalNumbers) return;

    long long num = start + tid * 2;  // we only test odd numbers

    bool isPrime = true;

    if (num <= 1) {
        isPrime = false;
    } else if (num == 2) {
        isPrime = true;
    } else if (num % 2 == 0) {
        isPrime = false;
    } else {
        for (long long i = 3; i * i <= num; i += 2) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }
    }

    // Write results into arrays
    numbers[tid]    = num;
    isPrimeArr[tid] = isPrime;
}

int main() {
    long long start = 100'001LL;  // must start with odd
    long long end   = 190'001LL;

    // How many odd numbers in [start, end]?
    int totalNumbers = (end - start) / 2 + 1;

    int threadsPerBlock = 256;
    int blocksPerGrid   = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    // Host arrays
    long long* h_numbers   = (long long*)malloc(totalNumbers * sizeof(long long));
    bool*      h_isPrime   = (bool*)malloc(totalNumbers * sizeof(bool));

    // Device arrays
    long long* d_numbers   = nullptr;
    bool*      d_isPrime   = nullptr;

    cudaMalloc((void**)&d_numbers, totalNumbers * sizeof(long long));
    cudaMalloc((void**)&d_isPrime, totalNumbers * sizeof(bool));

    // Timing setup
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    // Launch kernel
    checkPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(
        start,
        totalNumbers,
        d_numbers,
        d_isPrime
    );
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0.0f;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU kernel: " << gpuDuration << " ms" << std::endl;

    // Copy results back to host
    cudaMemcpy(h_numbers, d_numbers,
               totalNumbers * sizeof(long long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_isPrime, d_isPrime,
               totalNumbers * sizeof(bool),
               cudaMemcpyDeviceToHost);

    // Print only prime numbers
    std::cout << "Prime numbers in range [" << start << ", " << end << "]:" << std::endl;
    for (int i = 0; i < totalNumbers; ++i) {
        if (h_isPrime[i]) {
            std::cout << h_numbers[i] << " ";
        }
    }
    std::cout << std::endl;

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_numbers);
    cudaFree(d_isPrime);
    free(h_numbers);
    free(h_isPrime);

    return 0;
}
