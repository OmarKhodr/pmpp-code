#include "kernel.h"

#include "timer.h"
#include "cuda_check.h"

// Note: Tile dim MUST match block size in this implementation
constexpr int kTileDim = 32;

// Compute matrix multiplication (C = A * B)
// Dimensions: A (m*k), B (k*n), C (m*n)
// One thread = one output matrix element
__global__
void matMultTiledKernel(float *A, float *B, float *C, int m, int n, int k) {
  __shared__ float A_s[kTileDim][kTileDim];
  __shared__ float B_s[kTileDim][kTileDim];

	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	float res = 0.f;

	for (int tile = 0; tile < (k + kTileDim - 1)/kTileDim; ++tile) {
  	// Load tile to shared memory
    if (i < m && (tile*kTileDim + threadIdx.x) < k) {
      A_s[threadIdx.y][threadIdx.x] = A[i*k + tile*kTileDim + threadIdx.x];
    } else {
      A_s[threadIdx.y][threadIdx.x] = 0.f;
    }
    if ((tile*kTileDim + threadIdx.y) < k && j < n) {
      B_s[threadIdx.y][threadIdx.x] = B[(tile*kTileDim + threadIdx.y) * n + j];
    } else {
      B_s[threadIdx.y][threadIdx.x] = 0.f;
    }
    // Wait for all threads to finish loading shared tile before computing
    __syncthreads();
    // Compute with tile
    for (int l = 0; l < kTileDim; ++l) {
      res += A_s[threadIdx.y][l] * B_s[l][threadIdx.x];
    }
    // Wait for all threads to finish computing before loading next tile
    __syncthreads();
	}

	if (i < m && j < n) {
	  C[i*n + j] = res;
	}
}

void matMultTiledGpu(float *A_h, float *B_h, float *C_h, int m, int n, int k) {
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
	// M * N. For each dimension, we calculate the num of blocks by dividing it
	// by the number of threads per block, and take the ceiling of that.
	// x = columns (N), y = rows (M)
	dim3 num_blocks(
		(n + num_threads_per_block.x - 1) / num_threads_per_block.x,
		(m + num_threads_per_block.y - 1) / num_threads_per_block.y
	);
	matMultTiledKernel<<<num_blocks, num_threads_per_block>>>(
	  A_d, B_d, C_d, m, n, k);

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
