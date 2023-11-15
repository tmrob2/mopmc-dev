//
// Created by guoxin on 8/11/23.
//

namespace mopmc {
    namespace functions {
        namespace cuda {

            __global__ void aggregate(const double *w, const double *x, double *y, int numRows, int numObjs){
                // y = x * w
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < numRows) {
                    y[tid] = 0;
                    for (int i = 0; i < numObjs; ++i){
                        y[tid] += w[i] * x[i * numRows + tid];
                    }
                }
            }

            int aggregateLauncher(const double *w, const double *x, double *y, int numRows, int numObjs){
                int blockSize, minGridSize, gridSize;
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &aggregate, 0, numRows);
                gridSize = (numRows + blockSize - 1) / blockSize;
                aggregate<<<gridSize, blockSize>>>(w, x, y, numRows, numObjs);
                return 0;
            }

            __global__ void maxValue1(const double* y, double* x, const int* enabledActions,
                                      int* pi, int arrCount, int numRows) {
                // arrCount is the number of states in the model
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                if(tid < arrCount) {
                    // do some stuff
                    int actionStart = enabledActions[tid];
                    int actionEnd = enabledActions[tid+1];
                    /*
                    if(tid < arrCount - 1 ) {
                        actionEnd = enabledActions[tid + 1];
                    } else {
                        actionEnd = numRows;
                    }
                     */
                    int maxIndex = pi[tid];
                    double maxValue = y[actionStart + maxIndex];
                    //double maxValue1 = x[tid];
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

            int maxValueLauncher1(double *y, double *x, int *enabledActions, int* pi, int arrCount, int numRows){
                int blockSize;
                int minGridSize;
                int gridSize;

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &maxValue1, 0, arrCount);

                gridSize = (arrCount + blockSize - 1) / blockSize;

                maxValue1<<<gridSize, blockSize>>>(y, x, enabledActions, pi, arrCount, numRows);
                return 0;
            }

            __global__ void maxValue2(const double* y, double* x, const int* enabledActions,
                                      int* pi, int* bpi, int arrCount) {
                // arrCount is the number of states in the model
                uint tid = threadIdx.x + blockIdx.x * blockDim.x;
                if(tid < arrCount) {
                    // do some stuff
                    int actionStart = enabledActions[tid];
                    int actionEnd = enabledActions[tid+1];
                    int maxIndex = pi[tid];
                    double maxValue = y[actionStart + maxIndex];
                    //update pi and x
                    for (int action = 0; action < (actionEnd - actionStart); ++action ) {
                        if (y[actionStart + action] > maxValue) {
                            maxIndex = action;
                            maxValue = y[actionStart+action];
                        }
                    }
                    x[tid] = maxValue;
                    pi[tid] = maxIndex;

                    //update binary pi
                    for (int action = 0; action < (actionEnd - actionStart); ++action ) {
                        if (action == maxIndex) {
                            bpi[actionStart + action] = 1;
                        } else {
                            bpi[actionStart + action] = 0;
                        }
                    }
                }
            }

            int maxValueLauncher2(double *y, double *x, int *enabledActions, int* pi, int* bpi, int arrCount){
                int blockSize;
                int minGridSize;
                int gridSize;

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &maxValue2, 0, arrCount);

                gridSize = (arrCount + blockSize - 1) / blockSize;

                maxValue2<<<gridSize, blockSize>>>(y, x, enabledActions, pi, bpi, arrCount);
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