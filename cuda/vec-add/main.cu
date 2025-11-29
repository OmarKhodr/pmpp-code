#include <iostream>

#include "timer.h"

void vecAddCpu(float *A, float *B, float *C, int n) {
	for (int i = 0; i < n; ++i) {
		C[i] = A[i] + B[i];
	}
}

// Compute vector sum C = A + B
// Each thread performs one pair-wise addition
__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

void vecAddGpu(float *A_h, float *B_h, float *C_h, int n) {

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
	ko::Timer timer;

	// Allocate memory and initialize data
	unsigned int N = (argc > 1) ? atoi(argv[1]) : 32e6;
	float* a = (float*) malloc(N * sizeof(float));
	float* b = (float*) malloc(N * sizeof(float));
	float* c_cpu = (float*) malloc(N * sizeof(float));
	float* c_gpu = (float*) malloc(N * sizeof(float));
	for (int i = 0; i < N; ++i) {
		a[i] = rand();
		b[i] = rand();
	}

	// Compute on CPU
	timer.Start();
	vecAddCpu(a, b, c_cpu, N);
	timer.Stop();
	timer.Print("CPU Time", PrintColor::Cyan);

	// Compute on GPU
	timer.Start();
	vecAddGpu(a, b, c_gpu, N);
	timer.Stop();
	timer.Print("GPU Time", PrintColor::DarkGreen);

	// Verify correctness of result
	for (int i = 0; i < N; ++i) {
		float diff = c_cpu[i] - c_gpu[i];
		float tolerance = 1e9;
		if (diff > tolerance || -diff > tolerance) {
			std::cout << "Mismatch at index " << i << " ";
			std::cout << "(CPU Result = " << c_cpu[i] << ", GPU result = " << c_gpu[i] << ")";
			std::cout << std::endl;
			break;
		}
	}

	free(a);
	free(b);
	free(c_cpu);
	free(c_gpu);

	return 0;
}
