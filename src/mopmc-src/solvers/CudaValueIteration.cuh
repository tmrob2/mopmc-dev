//
// Created by guoxin on 15/11/23.
//

#ifndef MOPMC_CUDAVALUEITERATION_CUH
#define MOPMC_CUDAVALUEITERATION_CUH


#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

namespace mopmc {
    namespace value_iteration {
        namespace gpu {

            template<typename ValueType>
            class CudaValueIterationHandler {
            public:

                CudaValueIterationHandler(
                        const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                        const std::vector<int> &rowGroupIndices,
                        const std::vector<int> &row2RowGroupMapping,
                        std::vector<ValueType> &rho_flat,
                        std::vector<int> &pi,
                        int iniRow,
                        //std::vector<double> &w
                        int objCount);


                int initialise();

                int exit();

                int valueIteration(const std::vector<double> &w);

                int valueIterationPhaseOne(const std::vector<double> &w);

                int valueIterationPhaseTwo();

                Eigen::SparseMatrix<ValueType, Eigen::RowMajor> transitionMatrix_;
                std::vector<ValueType> flattenRewardVector_;
                std::vector<int> scheduler_;
                std::vector<int> rowGroupIndices_;
                std::vector<int> row2RowGroupMapping_;
                //std::vector<double> weightVector_;
                std::vector<double> weightedValueVector_;
                //std::vector<double> y_;
                std::vector<double> results_;
                double weightedResult_{};
                int iniRow_{};
                int nobjs{};

                const std::vector<double> &getResults() const {
                    return results_;
                }

            private:
                int *dA_csrOffsets{}, *dA_columns{}, *dA_rows_extra{};
                int *dB_csrOffsets{}, *dB_columns{}, *dB_rows_extra{};
                int *dRowGroupIndices{}, *dRow2RowGroupMapping{}, *dPi{};
                int *dMasking_nnz{}, *dMasking_nrows{}; // this is an array of 0s and 1s
                double *dA_values{}, *dB_values{};
                double *dR{};
                double *dW{}, *dRw{}, *dRi{}, *dRj;
                double *dX{}, *dXPrime{}, *dY{}, *dZ{}, *dZPrime{};
                double *dResult{};
                int A_nnz{}, A_ncols{}, A_nrows{};
                int B_nnz{}, B_ncols{}, B_nrows{};
                int C_ncols{}, C_nrows{}, C_ld{};

                double alpha=1.0, alpha2=-1.0, beta=1.0;
                double eps=1.0, maxEps=0.0;
                int maxIter=1000, maxInd = 0;
                int iteration{};

                //CUSPARSE APIs
                cublasHandle_t cublasHandle = nullptr;
                cusparseHandle_t handle = nullptr, handleB = nullptr;
                cusparseSpMatDescr_t matA{}, matB{};
                cusparseDnMatDescr_t matC{}, matD{};
                cusparseDnVecDescr_t vecRw{}, vecX{}, vecXPrime{}, vecY{};
                void *dBuffer = nullptr, *dBufferB = nullptr;
                size_t bufferSize = 0, bufferSizeB = 0;

            };

        }
    }
}

#endif //MOPMC_CUDAVALUEITERATION_CUH
