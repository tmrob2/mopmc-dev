#include "WarmUp.h"

namespace mopmc {
namespace kernels {
__global__ void waremupKernel() {
    // This kernel does minimal work to warmup the GPU
}

int launchWarmupKernel() {
    int numBlocks = 1;
    int numThreadsPerBlock = 1;

    // launch the kernel to warm up the GPU
    waremupKernel<<<numBlocks, numThreadsPerBlock>>>();

    // Ensure all GPU tasks are completed
    cudaDeviceSynchronize();

    return 0;
}
}
}


