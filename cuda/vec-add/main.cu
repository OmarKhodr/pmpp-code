#include <iostream>

#include "timer.h"

// Compute vector sum C = A + B
// Each thread performs one pair-wise addition
__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
	Timer timer;

	int size = n * sizeof(float);

	float *A_d, *B_d, *C_d;
	cudaMalloc((void**)&A_d, size);
	cudaMalloc((void**)&B_d, size);
	cudaMalloc((void**)&C_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	const int num_blocks = ceil(n/256.0);
	const int num_threads_per_block = 256;
	vecAddKernel<<<num_blocks, num_threads_per_block>>>(A_d, B_d, C_d, n);

	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

int main(int argc, const char *argv[]) {
	cudaDeviceSynchronize();

	std::cout << "Initializing..." << std::endl;

	// Allocate memory and initialize data
	unsigned int N = (argc > 1) ? atoi(argv[1]) : 32e6;
	float* a = (float*) malloc(N * sizeof(float));
	float* b = (float*) malloc(N * sizeof(float));
	float* c = (float*) malloc(N * sizeof(float));
	for (unsigned int i = 0; i < N; ++i) {
		a[i] = rand();
		b[i] = rand();
	}

	std::cout << "Calling vec add" << std::endl;

	vecAdd(a, b, c, N);

	std::cout << "Completed vec add" << std::endl;

	free(a);
	free(b);
	free(c);

	return 0;
}
