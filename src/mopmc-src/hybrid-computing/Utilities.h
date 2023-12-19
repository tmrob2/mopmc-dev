//
// Created by thomas on 05/11/23.
//

#ifndef MOPMC_UTILITIES_H
#define MOPMC_UTILITIES_H

#include <storm/utility/constants.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <Eigen/Sparse>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

namespace hybrid::utilities {

template<typename T>
class CuMDPMatrix {
public:
    CuMDPMatrix(Eigen::SparseMatrix<T, Eigen::RowMajor> const &transitionSystem,
                std::vector<int> const& rowGroupIndices);

    int initialiseMDPMatrix(const Eigen::SparseMatrix<T, Eigen::RowMajor> &transitionSystem,
                         std::vector<int> const& rowGroupIndices);

    int tearDownMDPMatrix();

    void weightedSolution(std::vector<T>& w);

private:
    int *dA_csrOffsets;
    int *dA_columns;
    int *dEnabledActions;
    int *dPi;
    int A_nnz;
    int A_nCols;
    int A_nRows;
    double *dA_values;
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusparseSpMatDescr_t matA;
    void* dBuffer;
    size_t bufferSize;
};

}
#endif //MOPMC_UTILITIES_H
