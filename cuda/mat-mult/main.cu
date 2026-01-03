#include <iostream>

#include "timer.h"
#include "kernel.h"

void matMultCpu(float *A, float *B, float *C, int m, int n, int k) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			float res = 0.f;
			for (int l = 0; l < k; ++l) {
				res += A[i * k + l] * B[l * n + j];
			}
			C[i * n + j] = res;
		}
	}
}

int main(int argc, const char *argv[]) {
	cudaDeviceSynchronize();
	ko::Timer timer;

	// Allocate memory and initialize data
	unsigned int M = (argc > 1) ? std::atoi(argv[1]) : 200;
	unsigned int N = (argc > 2) ? std::atoi(argv[2]) : 500;
	unsigned int K = (argc > 3) ? std::atoi(argv[3]) : 1000;
	float* a = (float*) malloc(M * K * sizeof(float));
	float* b = (float*) malloc(K * N * sizeof(float));

	float* c_cpu = (float*) malloc(M * N * sizeof(float));
	float* c_gpu = (float*) malloc(M * N * sizeof(float));
	for (int i = 0; i < M * K; ++i) {
		a[i] = rand() * 1.0 / RAND_MAX;
	}
	for (int i = 0; i < K * N; ++i) {
		b[i] = rand() * 1.0 / RAND_MAX;
	}

	// Compute on CPU
	timer.Start();
	matMultCpu(a, b, c_cpu, M, N, K);
	timer.Stop();
	timer.Print("CPU Time", ko::PrintColor::Cyan);

	// Compute on GPU
	timer.Start();
	matMultGpu(a, b, c_gpu, M, N, K);
	timer.Stop();
	timer.Print("GPU Time", ko::PrintColor::DarkGreen);

	// Verify correctness of result
	for (int i = 0; i < M * N; ++i) {
		float diff = c_cpu[i] - c_gpu[i];
		const float tolerance = 0.00001;
		if (diff > tolerance || -diff > tolerance) {
			std::cout << "Mismatch at index " << i << " ";
			std::cout << "(CPU Result = " << c_cpu[i] << ", GPU result = " << c_gpu[i] << ", Difference: " << diff << ")";
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
