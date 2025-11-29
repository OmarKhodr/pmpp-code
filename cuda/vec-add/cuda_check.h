#ifndef _CUDA_CHECK_H_
#define _CUDA_CHECK_H_

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#endif  // _CUDA_CHECK_H_
