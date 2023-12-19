//
// Created by guoxin on 15/11/23.
//

#include "CudaValueIteration.cuh"
#include "CuFunctions.h"
//#include <storm/storage/SparseMatrix.h>
//#include <Eigen/Sparse>
#include <cstdio>
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
            CudaValueIterationHandler<ValueType>::CudaValueIterationHandler(
                    const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                    const std::vector<int> &rowGroupIndices,
                    const std::vector<int> &row2RowGroupMapping,
                    std::vector<ValueType> &rho_flat,
                    std::vector<int> &pi,
                    int iniRow,
                    int objCount) :
                    transitionMatrix_(transitionMatrix), flattenRewardVector_(rho_flat), scheduler_(pi),
                    rowGroupIndices_(rowGroupIndices), row2RowGroupMapping_(row2RowGroupMapping),
                    iniRow_(iniRow), nobjs(objCount) {

                A_nnz = transitionMatrix_.nonZeros();
                A_ncols = transitionMatrix_.cols();
                A_nrows = transitionMatrix_.rows();
                B_ncols = A_ncols;
                B_nrows = B_ncols;
                C_nrows = B_ncols;
                C_ncols = nobjs;
                C_ld = C_nrows;
                results_.resize(nobjs+1);
                //Assertions
                assert(A_ncols == scheduler_.size());
                assert(flattenRewardVector_.size() == A_nrows * nobjs);
                assert(rowGroupIndices_.size() == A_ncols + 1);
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::initialise() {
                std::cout << ("____ CUDA INITIALIZING ____\n");
                // cudaMalloc CONSTANTS -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dA_csrOffsets, (A_nrows + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dA_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dA_rows_extra, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dR, A_nrows * nobjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRowGroupIndices, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRow2RowGroupMapping, A_nrows * sizeof(int)))
                // cudaMalloc Variables -------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double))) // TR These vectors are not used in efficient DTMC Matrix-Matrix computation
                                                                                             // For now going to waste mem anda reinit as new vectors 
                                                                                             // Probably best to pass a flag here on whether to create these or not as they will
                                                                                             // always be dynamically created in the matrix-matrix version of DTMC VI
                CHECK_CUDA(cudaMalloc((void **) &dXPrime, A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dPi, A_ncols * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dW, nobjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dRw, A_nrows * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dResult, (nobjs + 1) * sizeof(double)))
                // cudaMalloc PHASE B-------------------------------------------------------------
                CHECK_CUDA(cudaMalloc((void **) &dB_csrOffsets, (A_ncols + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dB_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dB_rows_extra, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nrows, A_nrows * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dMasking_nnz, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void **) &dRi, B_nrows * sizeof(double))) // TR TODO: extra allocated mem which we possibly won't use in the hybrid version
                CHECK_CUDA(cudaMalloc((void **) &dRj, nobjs * B_nrows * sizeof(double))) // TR TODO: ^ 
                CHECK_CUDA(cudaMalloc((void **) &dZ, nobjs * A_ncols * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void **) &dZPrime, nobjs * A_ncols * sizeof(double)))
                // cudaMemcpy -------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix_.outerIndexPtr(), (A_nrows + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix_.innerIndexPtr(), A_nnz * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix_.valuePtr(), A_nnz * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dR, flattenRewardVector_.data(), A_nrows * nobjs * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dRowGroupIndices, rowGroupIndices_.data(), (A_ncols + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dRow2RowGroupMapping, row2RowGroupMapping_.data(), A_nrows * sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(dPi, scheduler_.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice));
                // NOTE. Data for dW copied in VI phase B.
                //CHECK_CUDA(cudaMemset(dY, static_cast<double>(0.0), A_nrows * sizeof(double)))
                //-------------------------------------------------------------------------
                CHECK_CUSPARSE(cusparseCreate(&handle))
                CHECK_CUBLAS(cublasCreate_v2(&cublasHandle));
                // Create sparse matrices A in CSR format
                CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nrows, A_ncols, A_nnz,
                                                 dA_csrOffsets, dA_columns, dA_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
                // Crease dense matrix C
                //CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C_nrows, C_ncols, C_ld, dZ, CUDA_R_64F, CUSPARSE_ORDER_COL));
                // Crease dense matrix D
                //CHECK_CUSPARSE(cusparseCreateDnMat(&matD, C_nrows, C_ncols, C_ld, dZPrime, CUDA_R_64F, CUSPARSE_ORDER_COL));
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
                CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIteration(const std::vector<double> &w) {

                this->valueIterationPhaseOne(w);
                this->valueIterationPhaseTwo();
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseOne(const std::vector<double> &w, bool toHost) {
                std::cout << "____ VI PHASE ONE ____\n" ;
                CHECK_CUDA(cudaMemcpy(dW, w.data(), nobjs * sizeof(double), cudaMemcpyHostToDevice))
                mopmc::functions::cuda::aggregateLauncher(dW, dR, dRw, A_nrows, nobjs);

                iteration = 0;
                do {
                    // y = r
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_nrows, dRw, 1, dY, 1))
                    if (iteration == 0) {
                        mopmc::functions::cuda::maxValueLauncher1(dY, dX, dRowGroupIndices, dPi, A_ncols + 1, A_nrows);
                    }
                    // y = A.x + r (r = y)
                    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                    // x' = x
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXPrime, 1))
                    // x(s) = max_{a\in Act(s)} y(s,a), pi(s) = argmax_{a\in Act(s)} pi(s)
                    mopmc::functions::cuda::maxValueLauncher1(dY, dX, dRowGroupIndices, dPi, A_ncols + 1, A_nrows);
                    // x' = -1 * x + x'
                    CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dX, 1, dXPrime, 1))
                    // max |x'|
                    CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd))
                    CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost))
                    maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                    //maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
                    ++iteration;
                    //printf("___ VI PHASE ONE, ITERATION %i, maxEps %f\n", iteration, maxEps);
                } while (maxEps > 1e-5 && iteration < maxIter);

                if (iteration == maxIter) {
                    std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                }
                //std::cout << "terminated after " << iteration <<" iterations.\n";
                //copy result
                thrust::copy(thrust::device, dX + iniRow_, dX + iniRow_ + 1, dResult + nobjs);
                if(toHost) {
                    CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                }

                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo() {
                std::cout << "____ VI PHASE TWO ____\n";
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_extra,
                                                CUSPARSE_INDEX_BASE_ZERO));

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dPi, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz - 1,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz - 1,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_extra, dA_rows_extra + A_nnz - 1,
                                dMasking_nnz, dB_rows_extra, mopmc::functions::cuda::is_not_zero<int>());
                // @B_nnz: number of non-zero entries in the DTMC transition matrix
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz - 1,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_extra, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_extra, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO));
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

                // value iteration for all objectives
                // !! As gpu does the main work, we can use mult-threading to send as many
                // individual objective data to gpu as possible.
                for (int obj = 0; obj < nobjs; obj++) {
                    thrust::copy_if(thrust::device, dR + obj * A_nrows, dR + (obj + 1) * A_nrows - 1,
                                    dMasking_nrows, dRi, mopmc::functions::cuda::is_not_zero<double>());

                    iteration = 0;
                    do {
                        // x = ri
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dX, 1));
                        // initialise x' as ri too
                        if (iteration == 0) {
                            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dRi, 1, dXPrime, 1));
                        }
                        // x = B.x' + ri where x = ri
                        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &alpha, matB, vecXPrime, &beta, vecX, CUDA_R_64F,
                                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
                        // x' = -1 * x + x'
                        CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, B_ncols, &alpha2, dX, 1, dXPrime, 1));
                        // max |x'|
                        CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd));
                        // get maxEps
                        CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost));
                        maxEps = (maxEps >= 0) ? maxEps : -maxEps;
                        // x' = x
                        CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows, dX, 1, dXPrime, 1));

                        //printf("___ VI PHASE TWO, OBJECTIVE %i, ITERATION %i, maxEps %f\n", obj, iteration, maxEps);
                        ++iteration;

                    } while (maxEps > 1e-5 && iteration < maxIter);
                    if (iteration == maxIter) {
                        std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                    }
                    //std::cout << "objective " << obj  << " terminated after " << iteration << " iterations\n";
                    // copy results
                    thrust::copy(thrust::device, dX + iniRow_, dX + iniRow_ + 1, dResult + obj);

                }

                //-------------------------------------------------------------------------
                CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(results_.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::exampleValueIterationPhaseTwo_v2(int beginObj, int endObj) {
                // This is just a test case to make sure that the matrix machinery is doing what we want it to do


                // value iteration for all objectives
                // !! As gpu does the main work, we can use mult-threading to send as many
                // individual objective data to gpu as possible.
                //
                // TR: We should avoid this loop and group together windows of the R vector into a matrix
                // one way to do this is just by copying a portion of dR
                //
                int nObjs           = endObj + 1 - beginObj;
                int A_num_rows      = 4;
                int A_num_cols      = 4;
                int A_nnz           = 9;
                int B_num_rows      = 7;
                int B_num_cols      = 3;
                int ldb             = B_num_rows;
                int ldc             = A_num_rows; 
                int B_size          = ldb * B_num_cols;
                int C_size          = ldc * nObjs;
                int hA_csrOffsets[] = {0, 3, 4, 7, 9};
                int hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
                double hA_values[]  = { 1.0, 2.0, 3.0, 4.0, 5.0,
                                        6.0, 7.0, 8.0, 9.0 };
                // In this scenario treat B as the total rewards vector and we just want a portion of it 
                // to turn into a dense matrix
                double hB[]          = { 1., 3., 2., 3., 3., 1., 4.,
                                        5.0,  2., 6., 5., 7., 2., 8.,
                                        9., 8., 10., 1., 11., 3., 12. };
                double* hResult = (double*)malloc(C_size * sizeof(double));

                double alpha        = 1.0;
                double beta         = 0.0;
                int *dA_csrOffsets, *dA_columns, *dMask, *dTiledMask;
                double *dA_values, *dB, *dC, *dZ; // dZ is the result matrix

                // do the masking on the CPU size so we can see what we are doing
                int r2rGMap[] = {0, 0, 1, 1, 2, 2, 3};
                int rowGroupIndices[] = {0, 2, 4, 6, 7}; // in this routine we will never use index 4 but keeping for completeness
                int mask[] = {0, 0, 0, 0, 0, 0, 0};
                int* tiledMask = (int*)malloc(B_num_rows * nObjs * sizeof(int));
                int pi[] = {0, 0, 0, 0};

                for (int i = 0; i < B_num_rows; ++i) {
                    int index = r2rGMap[i];
                    int firstRowInGroup = rowGroupIndices[index];
                    int selectedAction = pi[index];
                    if (i == firstRowInGroup + selectedAction) {
                        mask[i] = 1;
                    }
                }

                // copy the rewards data and the masking data to the GPU. Perform the masking operation Thrust
                CHECK_CUDA(cudaMalloc((void**) &dB, B_size * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void**) &dMask, B_num_rows * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void**) &dTiledMask, B_num_rows * nObjs * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void**) &dZ, B_num_rows * nObjs * sizeof(double)))
                CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(double), cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dMask, mask, B_num_rows * sizeof(int),cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemset(dTiledMask, 0, B_num_rows * nObjs * sizeof(int)))
                // This should be a smaller ptr because it stores the compressed values based on the mask
                CHECK_CUDA(cudaMemset(dZ, 0, A_num_cols * nObjs * sizeof(double))) 
                // Once B has been copied over to the device we need to perform the masking operation
                // We need to tile the mask the number of times that 
                mopmc::functions::cuda::tilingLauncher(dMask, dTiledMask, nObjs, B_num_rows);
                // cp the data back to the host so that I can see that the mask has been tiled correctly
                CHECK_CUDA(cudaMemcpy(tiledMask, dTiledMask, B_num_rows * nObjs * sizeof(int), cudaMemcpyDeviceToHost))
                
                printf("Tiled Array: \n");
                for (int i = 0 ; i < B_num_rows * nObjs; ++i) {
                    printf("%d", tiledMask[i]);
                }
                printf("\n");
                // This is working correctly up to this point

                // Now copy over the values from the rewards vector for the first two objectives,
                // The rewards vector is structured in column first format so it is a matter of 
                // selecting the 0th index to the 2 * n_rows + 1th index
                // Now the question is can I do it in one operation
                thrust::copy_if(thrust::device, dB + B_num_rows * beginObj, dB + B_num_rows * (beginObj + nObjs), 
                                dTiledMask, dZ, mopmc::functions::cuda::is_not_zero<double>());

                // Now copy the data back to the host and print to see if it is indeed the operation results
                // we expect
                CHECK_CUDA(cudaMemcpy(hResult, dZ, C_size * sizeof(double), cudaMemcpyDeviceToHost))

                printf("Masking operation of hB\n");
                for(int i = 0; i < A_num_cols * nObjs; ++i) {
                    printf("%0.1f, ", hResult[i]);
                }
                printf("\n");

                // Device memory management
                CHECK_CUDA(cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void**) &dA_values, A_nnz * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void**) &dC, C_size* sizeof(double)))
                // Cp the data from the host to the device if necessary
                CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(double), cudaMemcpyHostToDevice))
                CHECK_CUDA(cudaMemset(dC, 0, C_size * sizeof(double)))

                cusparseHandle_t     handle     = NULL;
                cusparseSpMatDescr_t matA;
                cusparseDnMatDescr_t matB, matC;
                void*                dBuffer    = NULL;
                size_t               bufferSize = 0;
                CHECK_CUSPARSE(cusparseCreate(&handle))
                // create a matrix A in CSR format
                CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                                 dA_csrOffsets, dA_columns, dA_values,
                                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
                                                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))

                // Create the matrix in position B
                CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, nObjs, A_num_cols, dZ, 
                                                    CUDA_R_64F, CUSPARSE_ORDER_COL))
                
                // Create the matrix in position C
                CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, nObjs, ldc, dC, 
                                                    CUDA_R_64F, CUSPARSE_ORDER_COL))

                // Allocate the external buffer
                CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, 
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                        &alpha, matA, matB, &beta, matC, CUDA_R_64F, 
                                                        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
                CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )                                      
                CHECK_CUSPARSE(cusparseSpMM(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

                // try and do a dense matrix addition using the dense matrix pointers
                cublasHandle_t cublasHandle;
                cublasCreate(&cublasHandle);

                alpha = 1.0;
                beta = 1.0;

                // need to make a result vector
                // create
                double *dCPrime;
                CHECK_CUDA(cudaMalloc((void**) &dCPrime, C_size* sizeof(double)))
                CHECK_CUDA(cudaMemset(dCPrime, 0, C_size * sizeof(double)))

                CHECK_CUBLAS(cublasDgeam_64(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                            A_num_rows, A_num_cols, &alpha, dC, A_num_rows, &beta, 
                                            dCPrime, A_num_rows, dCPrime, A_num_rows))

                // Destroy the matrix vector descriptors
                //CHECK_CUSPARSE(cusparseDestroySpMat(matA))
                //CHECK_CUSPARSE(cusparseDestroyDnMat(matB)) 
                //CHECK_CUSPARSE(cusparseDestroyDnMat(matC)) 
                //CHECK_CUSPARSE(cusparseDestroy(handle)) 

                // Device result check
                CHECK_CUDA(cudaMemcpy(hResult, dC, C_size * sizeof(double), cudaMemcpyDeviceToHost))

                for(int i = 0; i < A_num_rows; i++) {
                    for (int j = 0; j < nObjs; j++) {
                        printf("%0.1f, ", hResult[i + j * ldc]);
                    }
                    printf("\n");
                }
            
                //CHECK_CUDA( cudaFree(dBuffer) )
                //CHECK_CUDA( cudaFree(dA_csrOffsets) )
                //CHECK_CUDA( cudaFree(dA_columns) )
                //CHECK_CUDA( cudaFree(dA_values) )
                CHECK_CUDA( cudaFree(dB) )
                CHECK_CUDA( cudaFree(dC) )
                free(hResult);
                free(tiledMask);

                //-------------------------------------------------------------------------
                //CHECK_CUDA(cudaMemcpy(scheduler_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost));
                //CHECK_CUDA(cudaMemcpy(results_.data(), dResult, (nobjs + 1) * sizeof(double), cudaMemcpyDeviceToHost));
                return EXIT_SUCCESS;
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo_v2(int beginObj, int endObj) {

                std::cout << "____ VI PHASE TWO ____\n";
                int nObjs = endObj + 1 - beginObj;
                // First thing to do is to make the mask for both a particular R objective vector and the nnz
                // for the MDP transition matrix
                // Call GS's matrix building operations on the device
                // generate a DTMC transition matrix as a csr matrix
                CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_extra,
                                                CUSPARSE_INDEX_BASE_ZERO));

                mopmc::functions::cuda::binaryMaskingLauncher(dA_csrOffsets,
                                                              dRowGroupIndices, dRow2RowGroupMapping,
                                                              dPi, dMasking_nrows, dMasking_nnz, A_nrows);
                thrust::copy_if(thrust::device, dA_values, dA_values + A_nnz - 1,
                                dMasking_nnz, dB_values, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_columns, dA_columns + A_nnz - 1,
                                dMasking_nnz, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
                thrust::copy_if(thrust::device, dA_rows_extra, dA_rows_extra + A_nnz - 1,
                                dMasking_nnz, dB_rows_extra, mopmc::functions::cuda::is_not_zero<int>());
                // @B_nnz: number of non-zero entries in the DTMC transition matrix
                B_nnz = (int) thrust::count_if(thrust::device, dMasking_nnz, dMasking_nnz + A_nnz - 1,
                                               mopmc::functions::cuda::is_not_zero<double>());
                mopmc::functions::cuda::row2RowGroupLauncher(dRow2RowGroupMapping, dB_rows_extra, B_nnz);
                CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_extra, B_nnz, B_nrows,
                                                dB_csrOffsets, CUSPARSE_INDEX_BASE_ZERO));
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
                                                 dB_csrOffsets, dB_columns, dB_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
                // Now we have a CSR matrix matB which should be a square DTMC transition matrix
                // based on the optimised scheduler from phase 1

                // Now we need to call the tiling operations
                // make a tiledMask ptr
                int* dTiledMask;
                double *dZ2, *dX_, *dY_, *dEps; // is the dense rewards matrix ptr
                CHECK_CUDA(cudaMalloc((void**) &dTiledMask, A_nrows * nObjs * sizeof(int)))
                CHECK_CUDA(cudaMalloc((void**) &dZ2, B_nrows * nObjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void**) &dX_, B_nrows * nObjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void**) &dY_, B_nrows * nObjs * sizeof(double)))
                CHECK_CUDA(cudaMalloc((void**) &dEps, B_nrows * nObjs * sizeof(double)))
                // Set the values of dX and dY to zero
                CHECK_CUDA(cudaMemset(dX_, 0, B_nrows * nObjs * sizeof(double)))
                CHECK_CUDA(cudaMemset(dY_, 0, B_nrows * nObjs * sizeof(double)))

                functions::cuda::tilingLauncher(dMasking_nrows, dTiledMask, nObjs, A_nrows);
                // Now copy the reward vector starting at the startObj * A_nrows and finishing at beginObj* A_nrows + (beginObj + nObj) * A_nrows
                thrust::copy_if(thrust::device, dR + A_nrows * beginObj, dR + A_nrows * (beginObj + nObjs), 
                                dTiledMask, dZ2, mopmc::functions::cuda::is_not_zero<double>());
                // make the dense reward matrix in position C of the matC = alpha * matA * matB + beta * matC; 
                // beta must be set to 1.0;

                // Sp Matrix - Dn Matrix multiplication Memory Management
                //cublasCreate(&cublasHandle);
                //
                // Create the matrix in position B
                CHECK_CUSPARSE(cusparseCreateDnMat(&matC, B_nrows, nObjs, B_nrows, dX_, 
                                                    CUDA_R_64F, CUSPARSE_ORDER_COL))
                
                // Create the matrix in position C
                CHECK_CUSPARSE(cusparseCreateDnMat(&matD, B_nrows, nObjs, B_nrows, dY_, 
                                                    CUDA_R_64F, CUSPARSE_ORDER_COL))

                // Pre-iteration computation of the buffer.
                // The first operation is to multiply the DTMC transition matrix with the dense matrix X
                // This is a sparse matrix operation performed with cusparse
                // Create an appropriate buffer for the matrix computation
                // Allocate the external buffer
                iteration = 0;
                CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, 
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                        &alpha, matB, matC, &beta, matD, CUDA_R_64F, 
                                                        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
                CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )   

                do {                                   
                    // 1. Perform the matrix computation matC = alpha . matA * mat B + beta * matC
                    beta = 0.;
                    alpha = 1.;
                    // if this is first iteration then we should not perform this multiplication as it will just result in zeroes
                    if (iteration > 0) {
                        CHECK_CUSPARSE(cusparseSpMM(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matB, matC, &beta, matD, CUDA_R_64F,
                            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
                    }
                    // 2. Add the rewards vector to Y
                    // Do this with cublas_v2 Y = alpha . op(R) + beta . op(Y)
                    // make a cublas handler to handle this operation -> this is an inplace matrix - matrix summation
                    beta = 1.0;
                    CHECK_CUBLAS(cublasDgeam_64(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                                B_nrows, nObjs, &alpha, dZ2, B_nrows, &beta, 
                                                dY_, B_nrows, dY_, B_nrows))  
                    // 3. step is to compute the difference between dX and dY
                    alpha = 1.0;
                    beta = -1.0;
                    CHECK_CUBLAS(cublasDgeam_64(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                                B_nrows, nObjs, &alpha, dY_, B_nrows, 
                                                &beta, dX_, B_nrows, dEps, B_nrows))
                    // 4. Epsilon computation: find the index of the maximum epsion value
                    CHECK_CUBLAS(cublasIdamax(cublasHandle, B_ncols * nObjs, dEps, 1, &maxInd));
                    // get maxEps
                    CHECK_CUDA(cudaMemcpy(&maxEps, dEps + maxInd - 1, sizeof(double), cudaMemcpyDeviceToHost));
                    maxEps = (maxEps >= 0) ? maxEps : -maxEps;

                    //printf("Eps: %0.5f\n", maxEps);

                    // 5. Set the values of X to be Y
                    CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, B_nrows * nObjs, dY_, 1, dX_, 1))

                    ++iteration;
                } while (maxEps > 1e-5 && iteration < maxIter);
                if (iteration == maxIter) {
                    std::cout << "[warning] loop exit after reaching maximum iteration number (" << iteration <<")\n";
                };

                std::cout << "objectives found: terminated after " << iteration << " iterations\n";
                // copy results
                // copy the initial row of the matrix dX_
                for (int k = 0; k < nObjs; k++) {
                    thrust::copy(thrust::device, dX_ + k * B_nrows + iniRow_, dX_ + k * B_nrows + iniRow_ + 1, dResult + k);
                }

                // Memory clean up
                // Need to kill dX_, dY_, dZ2, dEps
                CHECK_CUDA(cudaFree(dX_))
                CHECK_CUDA(cudaFree(dY_))
                CHECK_CUDA(cudaFree(dZ2))
                CHECK_CUDA(cudaFree(dEps))
            }

            template<typename ValueType>
            int CudaValueIterationHandler<ValueType>::exit() {
                // destroy matrix/vector descriptors
                CHECK_CUSPARSE(cusparseDestroySpMat(matA))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecXPrime))
                CHECK_CUSPARSE(cusparseDestroyDnVec(vecRw))
                CHECK_CUSPARSE(cusparseDestroySpMat(matB))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
                CHECK_CUSPARSE(cusparseDestroyDnMat(matD))
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
                CHECK_CUDA(cudaFree(dZ))
                CHECK_CUDA(cudaFree(dZPrime))
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