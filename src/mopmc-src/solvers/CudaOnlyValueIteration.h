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
                      const std::vector<int> &rowGroupIndices, std::vector<ValueType> &rho_flat,
                      std::vector<int> &pi,
                      std::vector<double> &w,
                      std::vector<double> &x,
                      std::vector<double> &y);

        int initialise();

        int exit();

        int valueIterationPhaseOne(const std::vector<double> &w);

        int valueIteration();

        Eigen::SparseMatrix<ValueType, Eigen::RowMajor> transitionMatrix_;
        std::vector<ValueType> rho_;
        std::vector<int> pi_;
        std::vector<int> enableActions_;
        //int numObjs_;
        std::vector<double> w_;
        std::vector<double> x_;
        std::vector<double> y_;
        //std::vector<double> res_;

    private:
        int *dA_csrOffsets, *dA_columns, *dEnabledActions, *dPi;
        double *dA_values, *dX, *dY, *dR, *dRw, *dW, *dXTemp, *dXPrime;
        int A_nnz, A_ncols, A_nrows, nobjs;

        double alpha;
        double beta;
        double eps;

        //CUSPARSE APIs
        cublasHandle_t cublasHandle = nullptr;
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY, vecRw;
        void *dBuffer = nullptr;
        size_t bufferSize = 0;

    };

}
#endif //MOPMC_CUDAONLYVALUEITERATION_H
