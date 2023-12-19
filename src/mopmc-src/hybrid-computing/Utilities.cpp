//
// Created by thomas on 05/11/23.
//
#include "Utilities.h"

namespace hybrid::utilities {

template<typename T>
CuMDPMatrix<T>::CuMDPMatrix(const Eigen::SparseMatrix<T, Eigen::RowMajor> &transitionSystem,
                            std::vector<int> const& rowGroupIndices) {
    auto initRes = initialiseMatrix(transitionSystem, rowGroupIndices);
}

template <typename T>
int CuMDPMatrix<T>::initialiseMDPMatrix(const Eigen::SparseMatrix<T, Eigen::RowMajor> &transitionSystem,
                                     std::vector<int> const& rowGroupIndices) {
    CHECK_CUDA(cudaMalloc((void**) &dA_csrOffsets, (transitionSystem.rows() + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_values, A_nnz * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**) &dEnabledActions, A_nCols * sizeof(int )))
    cusparseHandle = nullptr;
    cublasHandle = nullptr;
    dBuffer = nullptr;
    bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nRows, A_nCols, A_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
}

template<typename T>
int CuMDPMatrix<T>::tearDownMDPMatrix() {
    //-------------------------------------------------------------------------
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroy(cusparseHandle) )

    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    return EXIT_SUCCESS;
}

}
