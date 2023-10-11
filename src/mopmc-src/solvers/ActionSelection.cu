#include "ActionSelection.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>

namespace mopmc{
namespace kernels{

__global__ void print_kernel(const double printVal) {
    printf("Max eps: %f\n", printVal);
}

__global__ void max_reduction_kernel(double* y, double* x, int size, float *out)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = -INFINITY;
    if (i < size)
    {
        sdata[tid] = static_cast<float>(y[i] - x[i]);
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

__global__ void maxValue(const double* y, double* x, const int* enabledActions, int* pi, int arrCount) {
    // arrCount is the number of states in the model
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < arrCount) {
        // do some stuff
        int actionStart = enabledActions[tid];
        int actionEnd = enabledActions[tid + 1];
        double maxValue = x[tid];
        int maxIndex = pi[tid];
        for (int action = 0; action < (actionEnd - actionStart); ++action ) {
            if (y[actionStart + action] > maxValue) {
                maxIndex = action;
                maxValue = y[actionStart+action];
            }
        }
        x[tid] = maxValue;
        pi[tid] = maxIndex;
    }
}

int maxValueLauncher(double *y, double *x, int *enabledActions, int* pi, int arrCount){
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &maxValue, 0, arrCount);

    gridSize = (arrCount + blockSize - 1) / blockSize;

    maxValue<<<gridSize, blockSize>>>(y, x, enabledActions, pi, arrCount);
    return 0;
}

int launchPrintKernel(double printVal) {
    print_kernel<<<1, 1>>>(printVal);
    return 0;
}


struct minus_functor {
    //const double a;
    //minus_functor(double a_) : a( a_ ) {}

    __host__ __device__
    double operator()(const double& y, const double& x) const {
        return  y - x;
    }
};

// y = x, x = xTemp
double findMaxEps(double* y, int N, double maxDiff) {
    thrust::device_ptr<double> devY(y);
    maxDiff = *(thrust::max_element(devY, devY + N));
    return maxDiff;
    //std::cout << maxDiff << std::endl;
}



}
}