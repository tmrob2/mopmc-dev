//
// Created by guoxin on 8/11/23.
//

namespace mopmc {
    namespace functions {
        namespace cuda {
            __global__ void selectStateValues(const double *y, double *x, const int *stateActionIndices, int *pi, int arrCount) {
                // arrCount is the number of states in the model
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < arrCount) {
                    int stateAction = stateActionIndices[tid] + pi[tid];
                    x[tid] = y[stateAction];
                }
            }

            int selectStateValuesLauncher(double *y, double *x, int *stateActionIndices, int* pi, int arrCount){
                int blockSize;
                int minGridSize;
                int gridSize;

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &selectStateValues, 0, arrCount);

                gridSize = (arrCount + blockSize - 1) / blockSize;

                selectStateValues<<<gridSize, blockSize>>>(y, x, stateActionIndices, pi, arrCount);
                return 0;
            }

            __global__ void aggregate(const double *w, const double *x, double *z, int n, int m){
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < n) {
                    for (int i = 0; i < m; ++i){
                        z[tid] += w[tid] * x[i * n + tid];
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
        }
    }
}