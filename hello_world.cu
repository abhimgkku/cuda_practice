#include <cuda_runtime.h>
#include <iostream>
__global__ void hello_world(){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	printf("hello world Thread %d\n",idx);
	}

int main(){

	hello_world<<<1,10>>>();
	cudaDeviceSynchronize();
	return 0;
}
