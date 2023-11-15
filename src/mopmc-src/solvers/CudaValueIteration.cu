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

namespace mopmc { namespace value_iteration { namespace gpu {

    template<typename ValueType>
    CudaValueIterationHandler<ValueType>::CudaValueIterationHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                                                                    std::vector<ValueType> &rho_flat) :
            transitionMatrix_(transitionMatrix), flattenRewardVector(rho_flat) {}


    template<typename ValueType>
    CudaValueIterationHandler<ValueType>::CudaValueIterationHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                                                                    const std::vector<int> &rowGroupIndices,
                                                                    const std::vector<int> &row2RowGroupIndices,
                                                                    std::vector<ValueType> &rho_flat,
                                                                    std::vector<int> &pi,
                                                                    std::vector<double> &w,
                                                                    std::vector<double> &x,
                                                                    std::vector<double> &y) :
            transitionMatrix_(transitionMatrix), flattenRewardVector(rho_flat), scheduler_(pi),
            rowGroupIndices_(rowGroupIndices), row2RowGroupIndices_(row2RowGroupIndices),
            weightVector_(w), x_(x), y_(y) {
    }

    template<typename ValueType>
    int CudaValueIterationHandler<ValueType>::initialise(){

        A_nnz = transitionMatrix_.nonZeros();
        A_ncols = transitionMatrix_.cols();
        A_nrows = transitionMatrix_.rows();
        nobjs = weightVector_.size();
        //Assertions
        assert(A_ncols == x_.size());
        assert(A_ncols == scheduler_.size());
        assert(flattenRewardVector.size() == A_nrows * nobjs);
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
        CHECK_CUDA(cudaMalloc((void **) &dEnabledActions, (A_ncols+1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dW, nobjs * sizeof(double)))
        // cudaMalloc VARIABLES -------------------------------------------------------------
        CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dXPrime, A_ncols * sizeof(double)))
        //CHECK_CUDA(cudaMalloc((void **) &dXTemp, A_ncols * sizeof(double))) //TODO dXTemp not needed
        CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dPi, A_ncols * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dPi_bin, A_nrows * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dRw, A_nrows * sizeof(double)))
        // cudaMalloc B-------------------------------------------------------------
        CHECK_CUDA(cudaMalloc((void **) &dB_csrOffsets, (A_ncols + 1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dB_columns, A_nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dB_values, A_nnz * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dB_rows_extra, A_nnz * sizeof(int)))
        // cudaMemcpy -------------------------------------------------------------
        CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix_.outerIndexPtr(),
                              (A_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix_.innerIndexPtr(),
                              A_nnz * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix_.valuePtr(),
                              A_nnz * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dX, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dXPrime, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemcpy(dXTemp, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dY, y_.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemset(dY, static_cast<double>(0.0), A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMemcpy(dR, flattenRewardVector.data(), A_nrows * nobjs * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dEnabledActions, rowGroupIndices_.data(), (A_ncols + 1) * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dPi, scheduler_.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice))
        // NOTE. Data for dW in VI phase 1.
        //-------------------------------------------------------------------------
        CHECK_CUSPARSE(cusparseCreate(&handle))
        CHECK_CUBLAS(cublasCreate_v2(&cublasHandle));
        // Create a sparse matrix A in CSR format
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_nrows, A_ncols, A_nnz,
                                         dA_csrOffsets, dA_columns, dA_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
        // Create dense vector X
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_ncols, dX, CUDA_R_64F))
        // Create dense vector Y
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_nrows, dY, CUDA_R_64F))
        // Create dense vector Rw
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecRw, A_nrows, dRw, CUDA_R_64F))
        // allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
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
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecRw))
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
        CHECK_CUDA(cudaFree(dY))
        CHECK_CUDA(cudaFree(dR))
        CHECK_CUDA(cudaFree(dRw))
        CHECK_CUDA(cudaFree(dW))


        printf("____ CUDA EXIT!! ____\n");
        return EXIT_SUCCESS;
    }

    template<typename ValueType>
    int CudaValueIterationHandler<ValueType>::valueIterationPhaseOne(const std::vector<double> &w){
        printf("____THIS IS AGGREGATE FUNCTION____\n");
        CHECK_CUDA(cudaMemcpy(dW, w.data(), nobjs * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemset(dRw, static_cast<double>(0.0), A_nrows * sizeof(double)))
        mopmc::functions::cuda::aggregateLauncher(dW, dR, dRw, A_nrows, nobjs);
        int maxInd = 0;
        int iterations = 0;
        double alpha2 = -1.0;

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
            //mopmc::functions::cuda::maxValueLauncher1(dY, dX, dEnabledActions, dPi, A_ncols+1, A_nrows);
            mopmc::functions::cuda::maxValueLauncher2(dY, dX, dEnabledActions, dPi, dPi_bin, A_ncols + 1);

            // x' <- -1 * x + x'
            CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dX, 1, dXPrime, 1))

            // x(s) <- max_{a\in Act(s)} y(s,a), pi(s) <- argmax_{a\in Act(s)} pi(s)
            // max |x'|
            CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd))
            CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime+maxInd-1, sizeof(double), cudaMemcpyDeviceToHost))
            maxEps = (maxEps >= 0) ? maxEps : -maxEps;
            //maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
            //
            ++iterations;
            printf("___ VI PHASE ONE, ITERATION %i, maxEps %f\n", iterations, maxEps);
        } while (maxEps > 1e-5 && iterations < maxIter);

        return EXIT_SUCCESS;
    }

    template<typename ValueType>
    int CudaValueIterationHandler<ValueType>::valueIterationPhaseTwo() {

        CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA_csrOffsets, A_nnz, A_nrows, dA_rows_extra, CUSPARSE_INDEX_BASE_ZERO));

       // thrust::device_ptr< int > dvA_columns = thrust::device_pointer_cast(dA_columns);

        thrust::transform(thrust::device, dA_columns, dA_columns+A_nnz-1,
                          dPi_bin, dB_columns, thrust::multiplies<int>());
        thrust::transform(thrust::device, dA_rows_extra, dA_rows_extra+A_nnz-1,
                          dPi_bin, dB_rows_extra, thrust::multiplies<int>());

        thrust::copy_if(thrust::device, dA_values, dA_values+A_nnz-1,
                        dPi_bin, dB_values, mopmc::functions::cuda::is_not_zero<int>());
        thrust::copy_if(thrust::device, dA_columns, dA_columns+A_nnz-1,
                        dPi_bin, dB_columns, mopmc::functions::cuda::is_not_zero<int>());
        thrust::copy_if(thrust::device, dA_rows_extra, dA_rows_extra+A_nnz-1,
                        dPi_bin, dB_rows_extra, mopmc::functions::cuda::is_not_zero<int>());

        //std::cout << "XXXX: " << thrust::count_if(thrust::device,dA_values,dA_values+A_nnz-1, mopmc::functions::cuda::is_not_zero<double>())<<"\n";

        int B_nnz = (int) thrust::count_if(thrust::device,dB_values,dB_values+A_nnz-1,
                                           mopmc::functions::cuda::is_not_zero<double>());

        /*
        {thrust::device_ptr<double> out_prt = thrust::device_pointer_cast(dB_values);
            for (int i = 0; i < 1000; ++i) {std::cout << "dB_values[" <<i<< "]: " << *(out_prt+i);}
            printf("\n");
        }
         */
        int B_nrows = A_nrows;
        CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows_extra,B_nnz, B_nrows,
                                        dB_csrOffsets,CUSPARSE_INDEX_BASE_ZERO));
        for (int i = 0; i < nobjs; i++) {
            int iterations = 0;
            do {


                /*
                // Create a sparse matrix B in CSR format
                CHECK_CUSPARSE(cusparseCreateCsr(&matB, A_nrows, A_ncols, A_nnz,
                                                 dA_csrOffsets, dA_columns, dA_values,
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
                                                 */

            } while (maxEps > 1e-5 && iterations < maxIter);
        }

        return EXIT_SUCCESS;
    }



    template class CudaValueIterationHandler<double>;
} } }
