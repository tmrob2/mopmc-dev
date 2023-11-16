//
// Created by guoxin on 15/11/23.
//

#include "CudaValueIteration.cuh"

#include "CudaOnlyValueIteration.h"
#include "ActionSelection.h"
#include "CuFunctions.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <iostream>


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

namespace mopmc {
    namespace value_iteration {
        namespace gpu {

            template<typename ValueType>
            CudaValueIterationHandler<ValueType>::CudaValueIterationHandler(
                    const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                    std::vector<ValueType> &rho_flat) :
                    transitionMatrix_(transitionMatrix), flattenRewardVector_(rho_flat) {}


            template<typename ValueType>
            CudaValueIterationHandler<ValueType>::CudaValueIterationHandler(
                    const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                    const std::vector<int> &rowGroupIndices,
                    const std::vector<int> &row2RowGroupMapping,
                    std::vector<ValueType> &rho_flat,
                    std::vector<int> &pi,
                    std::vector<double> &w,
                    std::vector<double> &x,
                    std::vector<double> &y) :
                    transitionMatrix_(transitionMatrix), flattenRewardVector_(rho_flat), scheduler_(pi),
                    rowGroupIndices_(rowGroupIndices), row2RowGroupMapping_(row2RowGroupMapping),
                    weightVector_(w), x_(x), y_(y) {

                /*
                        {
                    int k = 1000;
                    printf("____ dRow2RowGroupMapping: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << row2RowGroupMapping_[i] << " ";
                    }
                    std::cout << "...]\n";
                }
                 */
            }


            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseOne(const std::vector<double> &w) {
                printf("____THIS IS AGGREGATE FUNCTION____\n");
                CHECK_CUDA(cudaMemcpy(dW, w.data(), nobjs * sizeof(double), cudaMemcpyHostToDevice))
                //CHECK_CUDA(cudaMemset(dRw, static_cast<double>(0.0), A_nrows * sizeof(double)))
                mopmc::functions::cuda::aggregateLauncher(dW, dR, dRw, A_nrows, nobjs);

                do {
                    /// y = r
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_nrows, dRw, 1, dY, 1))
                    // y = A.x + r (y = r)
                    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                    //
                    // compute the next policy update
                    // x' <- x
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXPrime, 1))

                    // x(s) <- max_{a\in Act(s)} y(s,a), pi(s) <- argmax_{a\in Act(s)} pi(s)
                    mopmc::functions::cuda::maxValueLauncher1(dY, dX, dRowGroupIndices, dPi, A_ncols + 1, A_nrows);

                    // x' <- -1 * x + x'
                    CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dX, 1, dXPrime, 1))

                    // x(s) <- max_{a\in Act(s)} y(s,a), pi(s) <- argmax_{a\in Act(s)} pi(s)
                    // max |x'|
                    CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd))
                    CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost))
                    maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                    //maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
                    //
                    ++iterations;
                    //printf("___ VI PHASE ONE, ITERATION %i, maxEps %f\n", iterations, maxEps);
                } while (maxEps > 1e-5 && iterations < maxIter);

                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo() {


                // cudaMalloc B-------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dB_csrOffsets, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dB_rows_extra, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nrows, A_nrows * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nnz, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRi, B_nrows * sizeof(double)))

                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_extra,
                                                CUSPARSE_INDEX_BASE_ZERO));

                mopmc::functions::cuda::maskingLauncher(dA_csrOffsets,
                                                        dRowGroupIndices, dRow2RowGroupMapping,
                                                        dPi, dMasking_nrows, dMasking_nnz, A_nrows);
                /*
                {
                    int k = 1000;
                    std::vector<int> hRowGroupIndices(A_nrows);
                    CHECK_CUDA(cudaMemcpy(hRowGroupIndices.data(), dRowGroupIndices, k * sizeof(int), cudaMemcpyDeviceToHost))
                    printf("____ dRowGroupIndices: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << hRowGroupIndices[i] << " ";
                    }
                    std::cout << "...]\n";
                }
                {
                    int k = 1000;
                    std::vector<int> hRow2RowGroupMapping(A_nrows);
                    CHECK_CUDA(cudaMemcpy(hRow2RowGroupMapping.data(), dRow2RowGroupMapping, k * sizeof(int), cudaMemcpyDeviceToHost))
                    printf("____ dRow2RowGroupMapping: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << hRow2RowGroupMapping[i] << " ";
                    }
                    std::cout << "...]\n";
                }
                {
                    int k = 1000;
                    std::vector<int> hPi(A_ncols);
                    CHECK_CUDA(cudaMemcpy(hPi.data(), dPi, k * sizeof(int), cudaMemcpyDeviceToHost))
                    printf("____ dPi: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << hPi[i] << " ";
                    }
                    std::cout << "...]\n";
                }
                {
                    int k = 1000;
                    std::vector<int> hMasking(A_nnz);
                    CHECK_CUDA(cudaMemcpy(hMasking.data(), dMasking_nnz, k * sizeof(int), cudaMemcpyDeviceToHost))
                    printf("____ dMasking_nnz: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << hMasking[i] << " ";
                    }
                    std::cout << "...]\n";
                }
                 */
                {
                    int k = 1000;
                    std::vector<int> hMaskingRows(A_nnz);
                    CHECK_CUDA(cudaMemcpy(hMaskingRows.data(), dMasking_nrows, k * sizeof(int), cudaMemcpyDeviceToHost))
                    printf("____ dMasking_nrows: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << hMaskingRows[i] << " ";
                    }
                    std::cout << "...]\n";
                }
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz - 1,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz - 1,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_extra, dA_rows_extra + A_nnz - 1,
                                dMasking_nnz, dB_rows_extra, mopmc::functions::cuda::is_not_zero<int>());
                B_nnz = (int) thrust::count_if(thrust::device, dB_values, dB_values + A_nnz - 1,
                                               mopmc::functions::cuda::is_not_zero<double>());
                /*
                {
                    int k = 100;
                    std::vector<int> hB_rows_extra(B_nnz);
                    CHECK_CUDA(cudaMemcpy(hB_rows_extra.data(), dB_rows_extra, B_nnz * sizeof(int), cudaMemcpyDeviceToHost))
                    printf("____ dB_rows_extra: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << hB_rows_extra[B_nnz - k + i] << " ";
                    }
                    std::cout << "...]\n";
                }
                 */
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_extra, B_nnz);
                /*
                {
                    int k = 100;
                    std::vector<int> hB_rows_extra(B_nnz);
                    CHECK_CUDA(cudaMemcpy(hB_rows_extra.data(), dB_rows_extra, B_nnz * sizeof(int), cudaMemcpyDeviceToHost))
                    printf("____ dB_rows_extra: [");
                    for (int i = 0; i < k; ++i) {
                        std::cout << hB_rows_extra[B_nnz - k + i] << " ";
                    }
                    std::cout << "...]\n";
                }
                 */
                std::cout << "A_ncols: " << A_ncols << ", A_nrows: " << A_nrows << ", A_nnz: " << A_nnz << "\n";
                std::cout << "B_ncols: " << B_ncols << ", B_nrows: " << B_nrows << ", B_nnz: " << B_nnz << "\n";

                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_extra,B_nnz, B_nrows,
                                               dB_csrOffsets,CUSPARSE_INDEX_BASE_ZERO));
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
                /*
               CHECK_CUSPARSE(cusparseCreateCoo(&matB, B_nrows, B_ncols, B_nnz,
                                                dB_rows_extra, dB_columns, dB_values,
                                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                CUDA_R_64F));
               */

                for (int i = 0; i < nobjs; i++) {

                    thrust::copy_if(thrust::device, dR+i*A_nrows, dR+(i+1)*A_nrows-1,
                                    dMasking_nrows, dRi, mopmc::functions::cuda::is_not_zero<double>());

                   /*
                    {
                        int k = 20;
                        std::vector<int> hR(A_nrows);
                        CHECK_CUDA(cudaMemcpy(hR.data(), dR, A_nrows * sizeof(double ), cudaMemcpyDeviceToHost))
                        printf("____ dR: [... ");
                        for (int i = 0; i < k; ++i) {
                            std::cout << hR[A_nrows - k + i] << " ";
                        }
                        std::cout << "]\n";
                    }

                    {
                        int k = 20;
                        std::vector<int> hRi(B_nrows);
                        CHECK_CUDA(cudaMemcpy(hRi.data(), dRi, B_nrows * sizeof(double ), cudaMemcpyDeviceToHost))
                        printf("____ dRi: [... ");
                        for (int i = 0; i < k; ++i) {
                            std::cout << hRi[B_nrows - k + i] << " ";
                        }
                        std::cout << "]\n";
                    }
                    */

                    int iterations2 = 0;
                    break;
                    do {
                        /// x' = r
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dXPrime, 1));
                        /// x = B.x + r (x' = r)
                        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &alpha, matB, vecXPrime, &beta, vecX, CUDA_R_64F,
                                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
                        // x' <- -1 * x + x'
                        CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, B_ncols, &alpha2, dX, 1, dXPrime, 1));
                        // max |x'|
                        CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd));
                        CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost));
                        maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                        //printf("___ VI PHASE TWO, ITERATION %i, maxEps %f\n", iterations2, maxEps);
                        ++iterations2;
                    } while (maxEps > 1e-5 && iterations2 < 100);
                }

                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::initialise() {

                A_nnz = transitionMatrix_.nonZeros();
                A_ncols = transitionMatrix_.cols();
                A_nrows = transitionMatrix_.rows();
                B_ncols = A_ncols;
                B_nrows = B_ncols;
                nobjs = weightVector_.size();
                //Assertions
                assert(A_ncols == x_.size());
                assert(A_ncols == scheduler_.size());
                assert(flattenRewardVector_.size() == A_nrows * nobjs);
                assert(rowGroupIndices_.size() == A_ncols + 1);

                alpha = 1.0;
                beta = 1.0;
                eps = 1.0;
                maxIter = 1000;
                maxEps = 0.0;

                // cudaMalloc CONSTANTS -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dA_csrOffsets, (A_nrows + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dA_rows_extra, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dR, A_nrows * nobjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRowGroupIndices, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRow2RowGroupMapping, A_nrows * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dW, nobjs * sizeof(double)))
                // cudaMalloc VARIABLES -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dXPrime, A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dPi, A_ncols * sizeof(int)))
                //CHECK_CUDA(cudaMalloc((void **) &dPi_bin, A_nrows * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRw, A_nrows * sizeof(double)))
                // cudaMemcpy -------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix_.outerIndexPtr(), (A_nrows + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix_.innerIndexPtr(), A_nnz * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix_.valuePtr(), A_nnz * sizeof(double),
                                      cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dX, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dXPrime, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dY, y_.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice));
                //CHECK_CUDA(cudaMemset(dY, static_cast<double>(0.0), A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMemcpy(dR, flattenRewardVector_.data(), A_nrows * nobjs * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dRowGroupIndices, rowGroupIndices_.data(), (A_ncols + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dRow2RowGroupMapping, row2RowGroupMapping_.data(), A_nrows * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dPi, scheduler_.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice));
                // NOTE. Data for dW in VI phase 1.
                //-------------------------------------------------------------------------
                CHECK_CUSPARSE(cusparseCreate(&handle))
                CHECK_CUBLAS(cublasCreate_v2(&cublasHandle));
                // Create sparse matrices A in CSR format
                CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nrows, A_ncols, A_nnz,
                                                 dA_csrOffsets, dA_columns, dA_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

                // Create dense vector X
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_ncols, dX, CUDA_R_64F))
                // Create dense vector Y
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_nrows, dY, CUDA_R_64F))
                // Create dense vector XPrime
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecXPrime, A_ncols, dXPrime, CUDA_R_64F))
                // Create dense vector Rw
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecRw, A_nrows, dRw, CUDA_R_64F))
                // allocate an external buffer if needed
                CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
                /*
                CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                        handleB, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matB, vecXPrime, &beta, vecX, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeB))
                        */
                CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
                //CHECK_CUDA(cudaMalloc(&dBufferB, bufferSizeB))
                //printf("____GOT HERE!!(@) ____\n");
                /////PRINTING
                std::vector<double> dXOut(A_ncols);
                CHECK_CUDA(cudaMemcpy(dXOut.data(), dX, A_ncols * sizeof(double), cudaMemcpyDeviceToHost))
                return EXIT_SUCCESS;
            }


            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::exit() {
                CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost))
                CHECK_CUDA(cudaMemcpy(x_.data(), dX, A_ncols * sizeof(double), cudaMemcpyDeviceToHost))
                //-------------------------------------------------------------------------
                // destroy matrix/vector descriptors
                CHECK_CUSPARSE(cusparseDestroySpMat(matA))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecXPrime))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecRw))
                CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                CHECK_CUSPARSE(cusparseDestroy(handle))

                // device memory de-allocation
                CHECK_CUDA(cudaFree(dBuffer))
                CHECK_CUDA(cudaFree(dA_csrOffsets))
                CHECK_CUDA(cudaFree(dA_columns))
                CHECK_CUDA(cudaFree(dA_values))
                CHECK_CUDA(cudaFree(dA_rows_extra))
                CHECK_CUDA(cudaFree(dB_csrOffsets))
                CHECK_CUDA(cudaFree(dB_columns))
                CHECK_CUDA(cudaFree(dB_values))
                CHECK_CUDA(cudaFree(dB_rows_extra))
                CHECK_CUDA(cudaFree(dX))
                CHECK_CUDA(cudaFree(dXPrime))
                CHECK_CUDA(cudaFree(dY))
                CHECK_CUDA(cudaFree(dR))
                CHECK_CUDA(cudaFree(dRw))
                CHECK_CUDA(cudaFree(dRi))
                CHECK_CUDA(cudaFree(dW))
                CHECK_CUDA(cudaFree(dRowGroupIndices))
                CHECK_CUDA(cudaFree(dRow2RowGroupMapping))
                CHECK_CUDA(cudaFree(dMasking_nrows))
                CHECK_CUDA(cudaFree(dMasking_nnz))


                printf("____ CUDA EXIT!! ____\n");
                return EXIT_SUCCESS;
            }

            template
            class CudaValueIterationHandler<double>;
        }
    }
}
