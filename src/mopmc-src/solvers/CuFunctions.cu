//
// Created by guoxin on 8/11/23.
//

namespace mopmc {
    namespace functions {
        namespace cuda {

            __global__ void aggregate(const double *w, const double *x, double *z, int n, int m){
                // z = x * w
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < n) {
                    z[tid] = 0;
                    for (int i = 0; i < m; ++i){
                        z[tid] += w[i] * x[i * n + tid];
                    }
                }
            }

            int aggregateLauncher(const double *w, const double *x, double *z, int k, int n, int m){
                int blockSize, minGridSize, gridSize;
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &aggregate, 0, k);
                gridSize = (m + blockSize - 1) / blockSize;
                aggregate<<<gridSize, blockSize>>>(w, x, z, n, m);
                return 0;
            }

            __global__ void maxValue(const double* y, double* x, const int* enabledActions, int* pi, int arrCount) {
                // arrCount is the number of states in the model
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                if(tid < arrCount) {
                    // do some stuff
                    int actionStart = enabledActions[tid];
                    int actionEnd = enabledActions[tid + 1];
                    double maxValue = y[actionStart + pi[tid]];
                    //double maxValue = x[tid];
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

            __global__ void abs(const double *x, int k) {
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                double diff = 0.0;
                if (tid < k) {
                    if (diff < x[tid] ) {
                        diff = x[tid];
                    }
                    if (diff < -1.0 * x[tid])
                        diff = x[tid];
                }
            }

            int absLauncher(const double *x, int k) {
                int blockSize, minGridSize, gridSize;
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &abs, 0, k);
                gridSize = (blockSize - 1) / blockSize;
                abs<<<gridSize, blockSize>>>(x, k);
                return 0;
            }

        }
    }
}