//
// Created by guoxin on 8/11/23.
//

#include "CudaOnlyValueIteration.h"
#include "ActionSelection.h"
#include "CuFunctions.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
//#include <thrust/device_ptr.h>
#include <iostream>
#include <storm/utility/vector.h>


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

namespace mopmc::value_iteration::cuda_only {

    template<typename ValueType>
    CudaIVHandler<ValueType>::CudaIVHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                                            std::vector<ValueType> &rho_flat) :
                                            transitionMatrix_(transitionMatrix), rho_(rho_flat) {}


    template<typename ValueType>
    CudaIVHandler<ValueType>::CudaIVHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                                            const std::vector<uint64_t> &rowGroupIndices,
                                            std::vector<ValueType> &rho_flat,
                                            std::vector<uint64_t> &pi,
                                            std::vector<double> &w,
                                            std::vector<double> &x,
                                            std::vector<double> &y) :
            transitionMatrix_(transitionMatrix), rho_(rho_flat), pi_(pi),
            enableActions_(rowGroupIndices), w_(w), x_(x), y_(y) {
    }

    template <typename ValueType>
    CudaIVHandler<ValueType>::CudaIVHandler(const Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionMatrix,
                  const std::vector<uint64_t> &rowGroupIndices,
                  const std::vector<uint_fast64_t>& ecqToOrigChoiceMapping_,
                  uint64_t origStateActions_,
                  std::vector<ValueType> &rho_flat,
                  std::vector<uint64_t> &pi,
                  std::vector<double> &w,
                  std::vector<double> &x,
                  std::vector<double> &y) :
        transitionMatrix_(transitionMatrix), pi_(pi),
        enableActions_(rowGroupIndices), ecqToOrigChoiceMapping(ecqToOrigChoiceMapping_),
        origStateActions(origStateActions_),
        w_(w), x_(x), y_(y) {
        rewardsDownScaling(rho_flat); //  Added a new downscaled rho

        std::cout << "rho size " << rho_.size() << " A_nrows * nobjs " << A_nrows * nobjs << "\n";
    }

    template <typename ValueType>
    void CudaIVHandler<ValueType>::rewardsDownScaling(std::vector<ValueType>& rho_flat) {

        auto nrows = transitionMatrix_.rows();
        auto nobjs = w_.size();
        rho_.resize(nrows * nobjs);
        for (uint_fast64_t i = 0 ; i < nobjs; ++i) {
            for (uint_fast64_t j = 0; j < nrows; ++j) {
                rho_[i * nrows + j] = rho_flat[i*origStateActions + ecqToOrigChoiceMapping[j]];
            }
        }
    }


    template<typename ValueType>
    int CudaIVHandler<ValueType>::initialise(){

        A_nnz = transitionMatrix_.nonZeros();
        A_ncols = transitionMatrix_.cols();
        std::cout << "A_ncols " << A_ncols << " x: " << x_.size() <<"\n";
        A_nrows = transitionMatrix_.rows();
        nobjs = w_.size();
        //Assertions
        assert(A_ncols == x_.size());
        assert(A_ncols == pi_.size());
        assert(rho_.size() == A_nrows * nobjs);
        assert(enableActions_.size() == A_ncols + 1);

        std::cout << "____ PRINTING RHO_: [ ";
        for (int i = 0; i < 50; ++i) {
            std::cout << rho_[i] << " ";
        } std::cout << "]\n";



        alpha = 1.0;
        beta = 1.0;
        eps = 1.0;

        // cudaMalloc part 1 -------------------------------------------------------------
        CHECK_CUDA(cudaMalloc((void **) &dA_csrOffsets, (A_nrows + 1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dA_values, A_nnz * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dXPrime, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dXTemp, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dR, A_nrows * nobjs * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dEnabledActions, (A_ncols+1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dPi, A_ncols * sizeof(uint)))
        // cudaMalloc part 2 -------------------------------------------------------------
        CHECK_CUDA(cudaMalloc((void **) &dW, nobjs * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dRw, A_nrows * sizeof(double)))
        //printf("____GOT HERE!!____\n");
        CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix_.outerIndexPtr(),
                              (A_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix_.innerIndexPtr(),
                              A_nnz * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix_.valuePtr(),
                              A_nnz * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dX, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dXPrime, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dXTemp, x_.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemset(dX, static_cast<double>(0.0), A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMemcpy(dY, y_.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemset(dY, static_cast<double>(0.0), A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMemcpy(dR, rho_.data(), A_nrows * nobjs * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemcpy(dRw, y_.data(), A_nrows * sizeof(double ), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dEnabledActions, enableActions_.data(), (A_ncols+1) * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dPi, pi_.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice))
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
        printf("____ dX (at the end of initialisation): [");
        for (int i = 0; i < 50; ++i) {
            std::cout << dXOut[i] << " ";
        }
        std::cout << "]\n" ;
        //

        return EXIT_SUCCESS;
    }


    template<typename ValueType>
    int CudaIVHandler<ValueType>::exit() {
        CHECK_CUDA(cudaMemcpy(pi_.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost))
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
        CHECK_CUDA(cudaFree(dX))
        CHECK_CUDA(cudaFree(dY))
        CHECK_CUDA(cudaFree(dR))
        CHECK_CUDA(cudaFree(dRw))
        CHECK_CUDA(cudaFree(dW))

        printf("____ CUDA EXIT!! ____\n");
        return EXIT_SUCCESS;
    }

    template<typename ValueType>
    int CudaIVHandler<ValueType>::valueIterationPhaseOne(const std::vector<double> &w){
        printf("____THIS IS AGGREGATE FUNCTION____\n");
        CHECK_CUDA(cudaMemcpy(dW, w.data(), nobjs * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemset(dRw, static_cast<double>(0.0), A_nrows * sizeof(double)))
        mopmc::functions::cuda::aggregateLauncher(dW, dR, dRw, A_nrows, nobjs);
        ///// PRINTING FOR DEBUG

        std::vector<double> dWOut0(nobjs);
        CHECK_CUDA(cudaMemcpy(dWOut0.data(), dW, nobjs * sizeof(double), cudaMemcpyDeviceToHost))
        printf("____ dW: [");
        for (int i = 0; i < nobjs; ++i) {
            std::cout << dWOut0[i] << " ";
        }
        std::cout << "]\n" ;


        std::vector<double> dRwOut0(A_nrows);
        CHECK_CUDA(cudaMemcpy(dRwOut0.data(), dRw, A_nrows * sizeof(double), cudaMemcpyDeviceToHost))
        printf("____ dRw: [");
        for (int i = 0; i < 100; ++i) {
            std::cout << dRwOut0[i] << " ";
        }
        std::cout << "]\n" ;

        /*
        std::vector<double> dROut(A_nrows * nobjs);
        CHECK_CUDA(cudaMemcpy(dROut.data(), dR, A_nrows * nobjs * sizeof(double), cudaMemcpyDeviceToHost))
        printf("____ dROut: [");
        for (int i = 0; i < 1000; ++i) {
            std::cout << dROut[i] << " ";
        }
        std::cout << "]\n" ;
         */
        /////
        double maxEps = 0.0;
        int maxInd = 0;
        int iterations = 0;
        double alpha2 = -1.0;
        int maxIter = 200;

        ////FOR PRINTING
        std::vector<double> dXOut(A_ncols);
        std::vector<double> dYOut(A_nrows);
        std::vector<double> dXPrimeOut(A_ncols);
        std::vector<double> dRwOut(A_nrows);

        do {
            // y = r
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_nrows, dRw, 1, dY, 1))
            // y = A.x + r (y = r)
            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
            //
            // compute the next policy update
            // x' <- x
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXPrime, 1))
            //
            ////PRINTING
            CHECK_CUDA(cudaMemcpy(dXOut.data(), dX, A_ncols* sizeof(double), cudaMemcpyDeviceToHost))
            printf("____ dX (before max value launcher): [");
            for (int i = 0; i < 100; ++i) {
                std::cout << dXOut[dXOut.size()-100+i] << " ";
            }
            std::cout << "... ]\n" ;
            CHECK_CUDA(cudaMemcpy(dYOut.data(), dY, A_nrows* sizeof(double), cudaMemcpyDeviceToHost))
            printf("____ dY (before max value launcher): [");
            for (int i = 0; i < 200; ++i) {
                std::cout << dYOut[dYOut.size()-200+i] << " ";
            }
            std::cout << "... ]\n" ;
            /*
            CHECK_CUDA(cudaMemcpy(dRwOut.data(), dRw, A_nrows * sizeof(double), cudaMemcpyDeviceToHost))
            printf("____ dRw: [");
            for (int i = 0; i < 100; ++i) {
                std::cout << dRwOut[i] << " ";
            }
            std::cout << "... ]\n" ;
             */
            // x(s) <- max_{a\in Act(s)} y(s,a), pi(s) <- argmax_{a\in Act(s)} pi(s)
            mopmc::functions::cuda::maxValueLauncher1(dY, dX, dEnabledActions, dPi, A_ncols+1, A_nrows);
            //
            CHECK_CUDA(cudaMemcpy(dXOut.data(), dX, A_ncols* sizeof(double), cudaMemcpyDeviceToHost))
            ////PRINTING
            printf("____ dX (after max value launcher): [");
            for (int i = 0; i < 100; ++i) {
                std::cout << dXOut[dXOut.size()-100+i] << " ";
            }
            std::cout << "... ]\n" ;
            // x' <- -1 * x + x'
            CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dX, 1, dXPrime, 1))
            CHECK_CUDA(cudaMemcpy(dXPrimeOut.data(), dXPrime, A_ncols * sizeof(double), cudaMemcpyDeviceToHost))
            printf("____ dXPrime (x'-x): [");
            for (int i = 0; i < 200; ++i) {
                std::cout << dXPrimeOut[i] << " ";
            }
            std::cout << "... ]\n" ;

            // x(s) <- max_{a\in Act(s)} y(s,a), pi(s) <- argmax_{a\in Act(s)} pi(s)
            // max |x'|
            // maxEps: Host variable that will store the maximum value.
            // Array maximum index (in FORTRAN base).
            // Call cublas to get maxIndex: note that maxIndex is passed as a pointer to the cublas call.
            CHECK_CUBLAS(cublasIdamax(cublasHandle, A_ncols, dXPrime, 1, &maxInd))
            // Copy max value onto host variable: variable must be passed as pointer.
            // We offset our array by the index returned by cublas. It is important to notice that
            // we must reduce also by one since this is FORTRAN based indexing.
            CHECK_CUDA(cudaMemcpy(&maxEps, dXPrime+maxInd-1, sizeof(double), cudaMemcpyDeviceToHost))
            // We are not done yet, since the value may be negative.
            maxEps = (maxEps >= 0) ? maxEps : -maxEps;
            printf("Absolute maximum value of array is %lf.\n", maxEps);
            //maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
            //
            ++iterations;
            printf("___ VI PHASE ONE, ITERATION %i, maxEps %f\n", iterations, maxEps);
        } while (maxEps > 1e-5 && iterations < maxIter);

        return EXIT_SUCCESS;
    }


    template<typename ValueType>
    int CudaIVHandler<ValueType>::valueIteration() {

        // Execute value iteration
        double maxEps = 0.0, max = 1.;
        int iterations = 0;
        double alpha2 = -1.0;
        int maxIter = 10000;
        do {
            // y = r
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_nrows, dR, 1, dY, 1))
            // y = A.x + r(y = r)
            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
            // compute the next policy update
            // This updates x with the maximum value associated with an action
            mopmc::kernels::maxValueLauncher(dY, dX, dEnabledActions, dPi, A_ncols);
            // In the following computation dXTemp will be changed to dXTemp = dX - dXTemp
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXPrime, 1))
            CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dXTemp, 1, dXPrime, 1))
            maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXTemp, 1))

            ++iterations;
            printf("Cuda value iteration: %i, maxEps: %f\n", iterations, maxEps);
        } while (maxEps > 1e-5 && iterations < maxIter);

        return EXIT_SUCCESS;
    }

    template class CudaIVHandler<double>;
}