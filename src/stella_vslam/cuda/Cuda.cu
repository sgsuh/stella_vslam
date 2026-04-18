#include <cuda_runtime.h>
#include <stella_vslam/cuda/Cuda.h>

namespace stella_vslam {
namespace cuda {
void deviceSynchronize() {
    checkCudaErrors(cudaDeviceSynchronize());
}
}
}