#include "kernel.h"

#include "timer.h"
#include "cuda_check.h"

// Compute vector sum C = A + B
// Each thread performs one pair-wise addition
__global__
void matMultKernel(float *A, float *B, float *C, int m, int n, int k) {
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= m || j >= n) return;

	float res = 0.f;
	for (int l = 0; l < k; ++l) {
		res += A[i * k + l] * B[l * n + j];
	}
	C[i * n + j] = res;
}

void matMultGpu(float *A_h, float *B_h, float *C_h, int m, int n, int k) {
	ko::Timer gpu_timer;

	// Allocate GPU memory
	gpu_timer.Start();

	int a_size = m * k * sizeof(float);
	int b_size = k * n * sizeof(float);
	int c_size = m * n * sizeof(float);

	float *A_d, *B_d, *C_d;
	CUDA_CHECK(cudaMalloc((void**)&A_d, a_size));
	CUDA_CHECK(cudaMalloc((void**)&B_d, b_size));
	CUDA_CHECK(cudaMalloc((void**)&C_d, c_size));

	CUDA_CHECK(cudaDeviceSynchronize());
	gpu_timer.Stop();
	gpu_timer.Print("Allocation time");

	// Copy data to GPU
	gpu_timer.Start();

	CUDA_CHECK(cudaMemcpy(A_d, A_h, a_size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(B_d, B_h, b_size, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaDeviceSynchronize());
	gpu_timer.Stop();
	gpu_timer.Print("Copy to GPU time");

	// Call kernel
	gpu_timer.Start();

	dim3 num_threads_per_block(32, 32);
	// We assign threads to cells of the *output matrix*, which has dimensions
	// M * N. We take the ceiling division of each block dim.
	dim3 num_blocks(
		(n + num_threads_per_block.x - 1) / num_threads_per_block.x,
		(m + num_threads_per_block.y - 1) / num_threads_per_block.y
	);
	matMultKernel<<<num_blocks, num_threads_per_block>>>(A_d, B_d, C_d, m, n, k);

	CUDA_CHECK(cudaGetLastError()); // catch launch errors
	CUDA_CHECK(cudaDeviceSynchronize()); // catch runtime errors in kernel
	gpu_timer.Stop();
	gpu_timer.Print("Kernel time", ko::PrintColor::Green);

	// Copy data from GPU
	gpu_timer.Start();

	CUDA_CHECK(cudaMemcpy(C_h, C_d, c_size, cudaMemcpyDeviceToHost));

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
