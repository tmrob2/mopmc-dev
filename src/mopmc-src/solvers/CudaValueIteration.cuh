//
// Created by guoxin on 15/11/23.
//

#ifndef MOPMC_CUDAVALUEITERATION_CUH
#define MOPMC_CUDAVALUEITERATION_CUH


#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

namespace mopmc { namespace value_iteration { namespace gpu {

    template<typename ValueType>
    class CudaValueIterationHandler {
    public:


        CudaValueIterationHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                                  std::vector<ValueType> &rho_flat);


        CudaValueIterationHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                                  const std::vector<int> &rowGroupIndices,
                                  const std::vector<int> &row2RowGroupIndices,
                                  std::vector<ValueType> &rho_flat,
                                  std::vector<int> &pi,
                                  std::vector<double> &w,
                                  std::vector<double> &x,
                                  std::vector<double> &y);

        int initialise();

        int exit();

        int valueIterationPhaseOne(const std::vector<double> &w);

        int valueIterationPhaseTwo();

        //int valueIteration();

        Eigen::SparseMatrix<ValueType, Eigen::RowMajor> transitionMatrix_;
        std::vector<ValueType> flattenRewardVector;
        std::vector<int> scheduler_;
        std::vector<int> rowGroupIndices_;
        std::vector<int> row2RowGroupIndices_;
        std::vector<double> weightVector_;
        std::vector<double> x_;
        std::vector<double> y_;
        //std::vector<double> res_;

    private:
        int *dA_csrOffsets, *dA_columns, *dA_rows_extra;
        int *dB_csrOffsets, *dB_columns, *dB_rows_extra;
        int *dEnabledActions, *dPi;
        int *dPi_bin; // this is an array of 0s and 1s
        double *dA_values, *dB_values, *dX, *dY, *dR, *dRw, *dW, *dXTemp, *dXPrime;
        int A_nnz, A_ncols, A_nrows, nobjs;
        int B_nnz, B_ncols, B_nrows;

        double alpha;
        double beta;
        double eps;
        int maxIter;
        double maxEps;

        //CUSPARSE APIs
        cublasHandle_t cublasHandle = nullptr;
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matA, matB;
        cusparseDnVecDescr_t vecX, vecY, vecRw;
        void *dBuffer = nullptr;
        size_t bufferSize = 0;

    };

} } }

#endif //MOPMC_CUDAVALUEITERATION_CUH
