#include <iostream>

#include "timer.h"

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)


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
	ko::Timer gpu_timer;

	// Allocate GPU memory
	gpu_timer.Start();

	int size = n * sizeof(float);

	float *A_d, *B_d, *C_d;
	CUDA_CHECK(cudaMalloc((void**)&A_d, size));
	CUDA_CHECK(cudaMalloc((void**)&B_d, size));
	CUDA_CHECK(cudaMalloc((void**)&C_d, size));

	CUDA_CHECK(cudaDeviceSynchronize());
	gpu_timer.Stop();
	gpu_timer.Print("Allocation time");

	// Copy data to GPU
	gpu_timer.Start();

	CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaDeviceSynchronize());
	gpu_timer.Stop();
	gpu_timer.Print("Copy to GPU time");

	// Call kernel
	gpu_timer.Start();

	const int num_blocks = ceil(n/256.0);
	const int num_threads_per_block = 256;
	vecAddKernel<<<num_blocks, num_threads_per_block>>>(A_d, B_d, C_d, n);

	CUDA_CHECK(cudaGetLastError()); // catch launch errors
	CUDA_CHECK(cudaDeviceSynchronize()); // catch runtime errors in kernel
	gpu_timer.Stop();
	gpu_timer.Print("Kernel time", ko::PrintColor::Green);

	// Copy data from GPU
	gpu_timer.Start();

	CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaDeviceSynchronize());
	gpu_timer.Stop();
	gpu_timer.Print("Copy from GPU time");

	// Free GPU memory
	gpu_timer.Start();

	CUDA_CHECK(cudaFree(A_d));
	CUDA_CHECK(cudaFree(B_d));
	CUDA_CHECK(cudaFree(C_d));

	CUDA_CHECK(cudaDeviceSynchronize());
	gpu_timer.Stop();
	gpu_timer.Print("Deallocation time");
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
	timer.Print("CPU Time", ko::PrintColor::Cyan);

	// Compute on GPU
	timer.Start();
	vecAddGpu(a, b, c_gpu, N);
	timer.Stop();
	timer.Print("GPU Time", ko::PrintColor::DarkGreen);

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
