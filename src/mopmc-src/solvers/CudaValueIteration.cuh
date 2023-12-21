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
#include "BaseValueIteration.h"
#include "../QueryData.h"

namespace mopmc {
    namespace value_iteration {
        namespace gpu {

            template<typename ValueType>
            class CudaValueIterationHandler : public mopmc::value_iteration::BaseVIHandler<ValueType>{
            public:

                explicit CudaValueIterationHandler(mopmc::QueryData<ValueType,int> *queryData);

                CudaValueIterationHandler(
                    const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                    const std::vector<int> &rowGroupIndices,
                    const std::vector<int> &row2RowGroupMapping,
                    std::vector<ValueType> &rho_flat,
                    std::vector<int> &pi,
                    int iniRow,
                    int objCount);


                int initialize() override;

                int exit() override;

                int valueIteration(const std::vector<double> &w) override;

                int valueIterationPhaseOne(const std::vector<double> &w, bool toHost=false);

                int valueIterationPhaseTwo();

                int valueIterationPhaseTwo_deprecated();

                //int valueIterationPhaseTwo_dev() {
                //    return valueIterationPhaseTwo_dev(0, this->nobjs);
                //}

                int valueIterationPhaseTwo_v2(int beginObj, int endObj);


                mopmc::QueryData<ValueType, int> *data;

                Eigen::SparseMatrix<ValueType, Eigen::RowMajor> transitionMatrix;
                std::vector<ValueType> flattenRewardVector;
                std::vector<int> scheduler;
                std::vector<int> rowGroupIndices;
                std::vector<int> row2RowGroupMapping;
                std::vector<double> weightedValueVector;

                
                std::vector<double> results;
                int iniRow{};
                int nobjs{};

                const std::vector<double> &getResults() const override {
                    return results;
                }

            private:
                int *dA_csrOffsets{}, *dA_columns{}, *dA_rows_backup{};
                int *dB_csrOffsets{}, *dB_columns{}, *dB_rows_backup{};
                int *dRowGroupIndices{}, *dRow2RowGroupMapping{}, *dScheduler{};
                int *dMasking_nnz{}, *dMasking_nrows{}, *dMasking_tiled{}; // this is an array of 0s and 1s
                double *dA_values{}, *dB_values{};
                double *dR{}, *dRi{}, *dRj{}, *dRPart{};
                double *dW{}, *dRw{};
                double *dX{}, *dX1{}, *dY{}, *dZ{}, *dZ1{};
                double *dResult{};
                int A_nnz{}, A_ncols{}, A_nrows{};
                int B_nnz{}, B_ncols{}, B_nrows{};
                int Z_ncols{}, Z_nrows{}, Z_ld{};

                double alpha=1.0, alpha2=-1.0, beta=1.0;
                double eps=1.0, maxEps{}, tolerance = 1.0e-8;
                int maxIter=2000, maxInd = 0;
                int iteration{};

                //CUSPARSE APIs
                cublasHandle_t cublasHandle = nullptr;
                cusparseHandle_t handle = nullptr, handleB = nullptr;
                cusparseSpMatDescr_t matA{}, matB{};
                cusparseDnMatDescr_t matZ{}, matZ1{};
                cusparseDnVecDescr_t vecRw{}, vecX{}, vecX1{}, vecY{};
                void *dBuffer = nullptr, *dBufferB = nullptr;
                size_t bufferSize = 0, bufferSizeB = 0;
            };
        }
    }
}

#endif //MOPMC_CUDAVALUEITERATION_CUH
