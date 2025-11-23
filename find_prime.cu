#include <cuda_runtime.h>
#include <iostream>

__global__ void checkPrimeKernel(long long start, long long end) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long num = start + tid * 2;
    bool isPrime = true;
    if (num > end) return;
    
    if (num <= 1){ isPrime = false; return;}
    if (num == 2) { isPrime = true; return; }
    if (num % 2 == 0) { isPrime = false; return; }
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            isPrime = false;
            break;  
        }

}
printf("tid=%d %lld is prime? %d\n", tid, num, isPrime);
}
int main() {
    long long start =  100'001LL; // must start with odd
    long long end   =  190'001LL;

    int threadPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1;
    int blocksPerGrid = (totalNumbers + threadPerBlock -1) /  threadPerBlock;
    cudaEvent_t startEvent , stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    checkPrimeKernel <<<blocksPerGrid,threadPerBlock>>>(start,end);
    cudaEventRecord(stopEvent,0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout<< "Time taken on GPU kernel: "<< gpuDuration << "ms" << std::endl;
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);



}
