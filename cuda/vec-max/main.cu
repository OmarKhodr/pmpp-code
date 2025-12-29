#include <iostream>

#include "timer.h"
#include "kernel.h"

void vecMaxCpu(float *A, float *B, float *C, int n) {
	for (int i = 0; i < n; ++i) {
		C[i] = A[i] + B[i];
	}
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
	vecMaxCpu(a, b, c_cpu, N);
	timer.Stop();
	timer.Print("CPU Time", ko::PrintColor::Cyan);

	// Compute on GPU
	timer.Start();
	vecMaxGpu(a, b, c_gpu, N);
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
