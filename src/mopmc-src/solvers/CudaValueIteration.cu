//
// Created by guoxin on 15/11/23.
//

#include "CudaValueIteration.cuh"
#include "CuFunctions.h"
#include "../solvers/WarmUp.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/count.h>
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
            CudaValueIterationHandler<ValueType>::CudaValueIterationHandler(QueryData<ValueType, int> *queryData) :
                    data(queryData),
                    transitionMatrix(queryData->transitionMatrix),
                    rowGroupIndices(queryData->rowGroupIndices),
                    row2RowGroupMapping(queryData->row2RowGroupMapping),
                    flattenRewardVector(queryData->flattenRewardVector),
                    scheduler(queryData->scheduler),
                    iniRow(queryData->initialRow),
                    nobjs(queryData->objectiveCount)
            {
                A_nnz = transitionMatrix.nonZeros();
                A_ncols = transitionMatrix.cols();
                A_nrows = transitionMatrix.rows();
                B_ncols = A_ncols;
                B_nrows = B_ncols;
                Z_nrows = B_ncols;
                Z_ncols = nobjs;
                Z_ld = Z_nrows;
                results.resize(nobjs + 1);
                //Assertions
                assert(A_ncols == scheduler.size());
                assert(flattenRewardVector.size() == A_nrows * nobjs);
                assert(rowGroupIndices.size() == A_ncols + 1);
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::initialize() {
                //GPU warm up
                mopmc::kernels::launchWarmupKernel();
                std::cout << ("____ CUDA INITIALIZING ____\n");
                // cudaMalloc CONSTANTS -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dA_csrOffsets, (A_nrows + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dA_rows_backup, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dR, A_nrows * nobjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRowGroupIndices, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRow2RowGroupMapping, A_nrows * sizeof(int)))
                // cudaMalloc Variables -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dX1, A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dScheduler, A_ncols * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dW, nobjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRw, A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dResult, (nobjs + 1) * sizeof(double)))
                // cudaMalloc PHASE B-------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dB_csrOffsets, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dB_rows_backup, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nrows, A_nrows * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nnz, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_tiled, Z_ncols * A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRi, B_nrows * sizeof(double))) // this is depreciated, not used in future
                CHECK_CUDA(cudaMalloc((void **) &dRPart, Z_ncols * Z_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dZ, Z_ncols * Z_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dZ1, Z_ncols * Z_nrows * sizeof(double)))
                // cudaMemcpy -------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix.outerIndexPtr(), (A_nrows + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix.innerIndexPtr(), A_nnz * sizeof(int),
                                      cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix.valuePtr(), A_nnz * sizeof(double),
                                      cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dR, flattenRewardVector.data(), A_nrows * nobjs * sizeof(double),
                                      cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dRowGroupIndices, rowGroupIndices.data(), (A_ncols + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dRow2RowGroupMapping, row2RowGroupMapping.data(), A_nrows * sizeof(int),
                                      cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dScheduler, scheduler.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice))
                //-------------------------------------------------------------------------
                CHECK_CUSPARSE(cusparseCreate(&handle))
                CHECK_CUBLAS(cublasCreate_v2(&cublasHandle))
                // Create sparse matrices A in CSR format
                CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nrows, A_ncols, A_nnz,
                                                 dA_csrOffsets, dA_columns, dA_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
                CHECK_CUSPARSE(cusparseCreateDnMat(&matZ, Z_nrows, Z_ncols, Z_ld, dZ, CUDA_R_64F, CUSPARSE_ORDER_COL))
                CHECK_CUSPARSE(cusparseCreateDnMat(&matZ1, Z_nrows, Z_ncols, Z_ld, dZ1, CUDA_R_64F, CUSPARSE_ORDER_COL))
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_ncols, dX, CUDA_R_64F))
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_nrows, dY, CUDA_R_64F))
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecX1, A_ncols, dX1, CUDA_R_64F))
                CHECK_CUSPARSE(cusparseCreateDnVec(&vecRw, A_nrows, dRw, CUDA_R_64F))
                // allocate an external buffer if needed
                CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                                       CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
                CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIteration(const std::vector<double> &w) {

                this->valueIterationPhaseOne(w);
                //this->valueIterationPhaseTwo_deprecated();
                this->valueIterationPhaseTwo();
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseOne(const std::vector<double> &w, bool toHost) {
                std::cout << "____ VI PHASE ONE ____\n" ;
                CHECK_CUDA(cudaMemcpy(dW, w.data(), nobjs * sizeof(double), cudaMemcpyHostToDevice))
                mopmc::functions::cuda::aggregateLauncher(dW, dR, dRw, A_nrows, nobjs);

                iteration = 0;
                do {
                    // Y = R
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_nrows, dRw, 1, dY, 1))
                    if (iteration == 0) {
                        mopmc::functions::cuda::maxValueLauncher1(dY, dX, dRowGroupIndices, dScheduler, A_ncols + 1, A_nrows);
                    }
                    // Y = A.X + Y (Y = R)
                    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                    // X1 = X
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dX1, 1))
                    // X(s) = max_{a\in Act(s)} Y(s,a), scheduler(s) = argmax_{a\in Act(s)} scheduler(s)
                    mopmc::functions::cuda::maxValueLauncher1(dY, dX, dRowGroupIndices, dScheduler, A_ncols + 1, A_nrows);
                    // X1 = -1 * X+ X1
                    CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dX, 1, dX1, 1))
                    // max |X1|
                    CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dX1, 1, &maxInd))
                    // to get maxEps, we must reduce also by one since this is FORTRAN based indexing.
                    CHECK_CUDA(cudaMemcpy(&maxEps, dX1 + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost))
                    maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                    //maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
                    ++iteration;
                    //printf("___ VI PHASE ONE, ITERATION %i, maxEps %f\n", iteration, maxEps);
                } while (maxEps > tolerance && iteration < maxIter);

                if (iteration == maxIter) {
                    std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                }
                //std::cout << "terminated after " << iteration <<" iterations.\n";
                //copy result
                thrust::copy(thrust::device, dX + iniRow, dX + iniRow + 1, dResult + nobjs);
                if(toHost) {
                    CHECK_CUDA(cudaMemcpy(scheduler.data(), dScheduler, A_ncols * sizeof(int), cudaMemcpyDeviceToHost))
                }

                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo() {
                std::cout << "____ VI PHASE TWO ____\n";
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_backup,
                                                CUSPARSE_INDEX_BASE_ZERO))

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dScheduler, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_backup, dA_rows_backup + A_nnz,
                                dMasking_nnz, dB_rows_backup, mopmc::functions::cuda::is_not_zero<int>());
                /* @param B_nnz: number of non-zero entries in the DTMC transition matrix */
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_backup, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_backup, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO))
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
                CHECK_CUSPARSE(cusparseSpMM_bufferSize(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matB, matZ1, &beta, matZ,
                        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSizeB))
                CHECK_CUDA(cudaMalloc(&dBufferB, bufferSizeB))
                CHECK_CUSPARSE(cusparseSpMM_preprocess(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matB, matZ1, &beta, matZ, CUDA_R_64F,
                        CUSPARSE_SPMM_ALG_DEFAULT, dBufferB))

                for (uint i = 0; i < nobjs; ++i) {
                    thrust::copy(thrust::device, dMasking_nrows, dMasking_nrows + A_nrows, dMasking_tiled + i * A_nrows);
                }
                thrust::copy_if(thrust::device, dR, dR + nobjs * A_nrows,
                                dMasking_tiled, dRPart, mopmc::functions::cuda::is_not_zero<double>());

                iteration = 0;
                do {
                    // Z = R'
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, Z_ncols * Z_nrows, dRPart, 1, dZ, 1))
                    // initialise Z1 as R' too
                    if (iteration == 0) {
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, Z_ncols * Z_nrows, dRPart, 1, dZ1, 1))
                    }
                    // Z = B.Z1 + Z, where Z = R'
                    CHECK_CUSPARSE(cusparseSpMM(
                            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matB, matZ1, &beta, matZ,
                            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBufferB))

                    // Z1 = -1 * Z + Z1
                    CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, Z_ncols * Z_nrows, &alpha2, dZ, 1, dZ1, 1))
                    // max |Z1|
                    CHECK_CUBLAS(cublasIdamax(cublasHandle, Z_ncols * Z_nrows, dZ1, 1, &maxInd))
                    // to get maxEps, we must reduce also by one since this is FORTRAN based indexing.
                    CHECK_CUDA(cudaMemcpy(&maxEps, dZ1 + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost))
                    maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                    // Z1 = Z
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, Z_ncols * Z_nrows, dZ, 1, dZ1, 1))
                    //printf("___ VI PHASE TWO, OBJECTIVE %i, ITERATION %i, maxEps %f\n", obj, iteration, maxEps);
                    ++iteration;

                } while (maxEps > tolerance && iteration < maxIter);
                if (iteration == maxIter) {
                    std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                }

                // copy results
                for (int obj = 0; obj < nobjs; ++obj) {
                    thrust::copy(thrust::device, dZ + iniRow + obj * Z_nrows, dZ + iniRow + 1 + obj * Z_nrows, dResult + obj);
                }
                //-------------------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(scheduler.data(), dScheduler, A_ncols * sizeof(int), cudaMemcpyDeviceToHost))
                CHECK_CUDA(cudaMemcpy(results.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost))
                CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                CHECK_CUDA(cudaFree(dBufferB))
                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            [[deprecated]] int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo_deprecated() {
                std::cout << "____ VI PHASE TWO (deprecated) ____\n";
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_backup,
                                                CUSPARSE_INDEX_BASE_ZERO))

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dScheduler, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_backup, dA_rows_backup + A_nnz,
                                dMasking_nnz, dB_rows_backup, mopmc::functions::cuda::is_not_zero<int>());
                /* @param B_nnz: number of non-zero entries in the DTMC transition matrix */
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_backup, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_backup, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO))
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))

                // value iteration for all objectives
                // !! As gpu does the main work, we can use mult-threading to send as many
                // individual objective data to gpu as possible.
                for (int obj = 0; obj < nobjs; obj++) {
                    thrust::copy_if(thrust::device, dR + obj * A_nrows, dR + (obj + 1) * A_nrows,
                                    dMasking_nrows, dRi, mopmc::functions::cuda::is_not_zero<double>());

                    iteration = 0;
                    do {
                        // x = ri
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dX, 1))
                        // initialise x' as ri too
                        if (iteration == 0) {
                            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dX1, 1))
                        }
                        // x = B.x' + ri where x = ri
                        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &alpha, matB, vecX1, &beta, vecX, CUDA_R_64F,
                                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                        // x' = -1 * x + x'
                        CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, B_ncols, &alpha2, dX, 1, dX1, 1))
                        // max |x'|
                        CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dX1, 1, &maxInd))
                        // to get maxEps, we must reduce also by one since this is FORTRAN based indexing.
                        CHECK_CUDA(cudaMemcpy(&maxEps, dX1 + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost))
                        maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                        // x' = x
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dX, 1, dX1, 1))

                        //printf("___ VI PHASE TWO, OBJECTIVE %i, ITERATION %i, maxEps %f\n", obj, iteration, maxEps);
                        ++iteration;

                    } while (maxEps > tolerance && iteration < maxIter);
                    if (iteration == maxIter) {
                        std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                    }
                    //std::cout << "objective " << obj  << " terminated after " << iteration << " iterations\n";
                    // copy results
                    thrust::copy(thrust::device, dX + iniRow, dX + iniRow + 1, dResult + obj);
                }

                //-------------------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(scheduler.data(), dScheduler, A_ncols * sizeof(int), cudaMemcpyDeviceToHost))
                CHECK_CUDA(cudaMemcpy(results.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost))
                CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                return EXIT_SUCCESS;
            }

            /* GS: I commented this out after adopting the idea.
             * But I don't include beginObj and endObj.
             * The two parameters make more sense if we implement a hybrid computing manager.
            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo_v2(int beginObj, int endObj) {
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_backup,
                                                CUSPARSE_INDEX_BASE_ZERO));

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dP, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_backup, dA_rows_backup + A_nnz,
                                dMasking_nnz, dB_rows_backup, mopmc::functions::cuda::is_not_zero<int>());
                // @param B_nnz: number of non-zero entries in the DTMC transition matrix
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_backup, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_backup, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO));
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

                // value iteration for all objectives
                // !! As gpu does the main work, we can use mult-threading to send as many
                // individual objective data to gpu as possible.
                //
                // TR: We should avoid this loop and group together windows of the R vector into a matrix
                // one way to do this is just by copying a portion of dR
                //
                double* dRPortion, *dMaskTiled;
                size_t bufferSizeMM;
                void *dBufferMM = nullptr;
                CHECK_CUDA(cudaMalloc((void**) &dRPortion, (endObj - beginObj) * B_nrows * sizeof(double)))
                // also tile dMasking_nrows (endObj - beginObj) times
                CHECK_CUDA(cudaMalloc((void**) &dMaskTiled, (endObj - beginObj) * A_nrows * sizeof(double)))
                for (uint i = 0; i < (endObj - beginObj); ++i) {
                    thrust::copy(thrust::device, dMasking_nrows, dMasking_nrows + A_nrows, dMaskTiled + i * A_nrows);
                }
                // create a mask of dR based on the tiled dMasking rows starting at the objective index
                thrust::copy_if(thrust::device, dR + beginObj * A_nrows, dR + endObj * A_nrows,
                                dMaskTiled, dRPortion, mopmc::functions::cuda::is_not_zero<double>());
                // Make dRPortion into a dense matrix
                // create a new cuBlas handle => this can be preallocated todo
                double *dY2, *dX2; // TODO rename using proper convention -> insert into initialise once working
                CHECK_CUDA(cudaMalloc((void**) &dY2, (endObj - beginObj) * B_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void**) &dX2, (endObj - beginObj) * B_nrows * sizeof(double)))
                CHECK_CUDA(cudaMemset(dX2, 0, (endObj - beginObj) * B_nrows * sizeof(double)))
                cusparseDnMatDescr_t matY2, matX2; //descrR
                // Create the dense matrices with the cusparse dense api
                CHECK_CUSPARSE(cusparseCreateDnMat(&matY2, B_nrows, (endObj - beginObj), B_nrows,
                                                  dY2, CUDA_R_64F, CUSPARSE_ORDER_COL))
                CHECK_CUSPARSE(cusparseCreateDnMat(&matX2, B_nrows, (endObj - beginObj), B_nrows,
                                                  dX2, CUDA_R_64F, CUSPARSE_ORDER_COL))

                // set the values of the dense matrix dC to vector dRPortion
                CHECK_CUSPARSE(cusparseDnMatSetValues(matY2, dRPortion))
                // copy the values of dRPortion whihc is an (endObj - beginOb) * A+ncols
                // in this computation we want to use the kernel formulation
                // C = alpha * op(A) . op(B) + beta * C 
                CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matX2, &beta, matY2, 
                        CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSizeMM))

                CHECK_CUDA(cudaMalloc(&dBufferMM, bufferSizeMM));

                CHECK_CUSPARSE(cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                               &alpha, matA, matX2, &beta, matY2, CUDA_R_64F, 
                               CUSPARSE_SPMM_ALG_DEFAULT, &dBufferMM))

                CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matX2, &beta, matY2, CUDA_R_64F, 
                               CUSPARSE_SPMM_ALG_DEFAULT, &dBufferMM))
                ///---///
                // clean up
                CHECK_CUSPARSE(cusparseDestroyDnMat(matX2))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matY2))
                CHECK_CUDA(cudaFree(dRPortion))
                CHECK_CUDA(cudaFree(dMaskTiled))
                CHECK_CUDA(cudaFree(dBufferMM))
                //-------------------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(scheduler_.data(), dP, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(results_.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                return EXIT_SUCCESS;
            }
             */


            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::exit() {

                // destroy matrix/vector descriptors
                CHECK_CUSPARSE(cusparseDestroySpMat(matA))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecX1))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecRw))
                //CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matZ))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matZ1))
                CHECK_CUSPARSE(cusparseDestroy(handle))
                // device memory de-allocation
                CHECK_CUDA(cudaFree(dBuffer))
                CHECK_CUDA(cudaFree(dA_csrOffsets))
                CHECK_CUDA(cudaFree(dA_columns))
                CHECK_CUDA(cudaFree(dA_values))
                CHECK_CUDA(cudaFree(dA_rows_backup))
                CHECK_CUDA(cudaFree(dB_csrOffsets))
                CHECK_CUDA(cudaFree(dB_columns))
                CHECK_CUDA(cudaFree(dB_values))
                CHECK_CUDA(cudaFree(dB_rows_backup))
                CHECK_CUDA(cudaFree(dX))
                CHECK_CUDA(cudaFree(dX1))
                CHECK_CUDA(cudaFree(dY))
                CHECK_CUDA(cudaFree(dZ))
                CHECK_CUDA(cudaFree(dZ1))
                CHECK_CUDA(cudaFree(dR))
                CHECK_CUDA(cudaFree(dRw))
                CHECK_CUDA(cudaFree(dRi))
                CHECK_CUDA(cudaFree(dW))
                CHECK_CUDA(cudaFree(dRowGroupIndices))
                CHECK_CUDA(cudaFree(dRow2RowGroupMapping))
                CHECK_CUDA(cudaFree(dMasking_nrows))
                CHECK_CUDA(cudaFree(dMasking_nnz))
                CHECK_CUDA(cudaFree(dResult))

                std::cout << ("____ CUDA EXIT ____\n");
                return EXIT_SUCCESS;
            }

            template class CudaValueIterationHandler<double>;
        }
    }
}


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
 */