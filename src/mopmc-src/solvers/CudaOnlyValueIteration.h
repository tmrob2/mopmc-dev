//
// Created by guoxin on 8/11/23.
//

#ifndef MOPMC_CUDAONLYVALUEITERATION_H
#define MOPMC_CUDAONLYVALUEITERATION_H

#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

namespace mopmc::value_iteration::cuda_only {

    template<typename ValueType>
    class CudaIVHandler {
    public:


        CudaIVHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                      std::vector<ValueType> &rho_flat);

        CudaIVHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                      std::vector<uint64_t> &rowGroupIndices, std::vector<ValueType> &rho_flat,
                      std::vector<uint64_t> &pi, uint64_t numObjs);

        int initialise();

        int valueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> const &transitionMatrix,
                           std::vector<ValueType> &x,
                           std::vector<ValueType> &r,
                           std::vector<int> &pi,
                           std::vector<int> const &rowGroupIndices);

        /*
        int valueIteration(Eigen::SparseMatrix<V, Eigen::RowMajor> const &transitionMatrix,
                           std::vector<std::vector<V>> &R,
                           std::vector<V> &w,
                           std::vector<int> &pi,
                           std::vector<int> const &rowGroupIndices);

         */
        Eigen::SparseMatrix<ValueType, Eigen::RowMajor> transitionMatrix_;
        std::vector<ValueType> rho_;
        std::vector<uint64_t> pi_;
        std::vector<uint64_t> stateIndices_;
        uint64_t numObjs_;

    private:
        int *dA_csrOffsets, *dA_columns, *dEnabledActions, *dPi;
        double *dA_values, *dX, *dY, *dR, *dXTemp, *dXPrime;
        int A_nnz, A_ncols, A_nrows;

        double alpha = 1.0;
        double beta = 1.0;
        double eps = 1.0;

        //CUSPARSE APIs
        cublasHandle_t cublasHandle = nullptr;
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY, vecR;
        void *dBuffer = nullptr;
        size_t bufferSize = 0;

    };
}
#endif //MOPMC_CUDAONLYVALUEITERATION_H
