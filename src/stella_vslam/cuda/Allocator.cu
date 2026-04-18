#include <cassert>
#include <cuda_runtime.h>
#include <stella_vslam/cuda/Allocator.h>
#include <stella_vslam/cuda/Cuda.h>

namespace stella_vslam {
namespace cuda {
size_t Allocator::getPitch(size_t widthSize) {
    return 128 + widthSize - widthSize % 128;
}

bool Allocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) {
    if(rows > 1 && cols > 1) {
        mat->step = getPitch(elemSize * cols);
        checkCudaErrors(cudaMallocManaged(&mat->data, mat->step * rows));
    } else {
        checkCudaErrors(cudaMallocManaged(&mat->data, elemSize * cols * rows));
        mat->step = elemSize * cols;
    }

    mat->refcount = (int*)new int();

    return true;
}

void Allocator::free(cv::cuda::GpuMat* mat) {
    checkCudaErrors(cudaFree(mat->datastart));
    delete mat->refcount;
}

cv::cuda::GpuMat::Allocator* gpu_mat_allocator;
}
}

namespace {
using namespace stella_vslam;

void __attribute__((constructor)) init() {
    cuda::gpu_mat_allocator = new cuda::Allocator();
}
}