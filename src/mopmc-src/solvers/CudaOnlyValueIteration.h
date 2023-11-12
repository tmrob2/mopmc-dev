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
                      const std::vector<uint64_t> &rowGroupIndices, std::vector<ValueType> &rho_flat,
                      std::vector<uint64_t> &pi,
                      std::vector<double> &w,
                      std::vector<double> &x,
                      std::vector<double> &y);

        CudaIVHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                      const std::vector<uint64_t> &rowGroupIndices,
                      const std::vector<uint_fast64_t>& ecqToOrigChoiceMapping,
                      uint64_t origStateActions,
                      std::vector<ValueType> &rho_flat,
                      std::vector<uint64_t> &pi,
                      std::vector<double> &w,
                      std::vector<double> &x,
                      std::vector<double> &y);

        int initialise();

        int exit();

        int valueIterationPhaseOne(const std::vector<double> &w);

        // Extra helper function to help selecting indices for ecQuotient to downscale rhoFlat
        void rewardsDownScaling(std::vector<ValueType>& rho_flat);

        int valueIteration();

        Eigen::SparseMatrix<ValueType, Eigen::RowMajor> transitionMatrix_;
        std::vector<ValueType> rho_;
        std::vector<uint64_t> pi_;
        std::vector<uint64_t> enableActions_;
        uint64_t numObjs_;
        std::vector<double> w_;
        std::vector<double> x_;
        std::vector<double> y_;
        std::vector<double> res_;

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

        // Extra weight vector processing bit for handling ecQuotient
        std::vector<uint_fast64_t> ecqToOrigChoiceMapping;
        uint64_t origStateActions;

    };

}
#endif //MOPMC_CUDAONLYVALUEITERATION_H
