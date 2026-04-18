#pragma once

#include <cstdio>
#include <cstdlib>

namespace stella_vslam {
namespace cuda {

#define checkCudaErrors(val) do { \
cudaError_t err = (val); \
if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d - %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
} \
} while (0)

void deviceSynchronize();
}
}