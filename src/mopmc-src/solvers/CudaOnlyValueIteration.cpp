//
// Created by guoxin on 8/11/23.
//

#include "CudaOnlyValueIteration.h"
#include "ActionSelection.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
//#include <thrust/device_ptr.h>


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
            std::vector<uint64_t> &rowGroupIndices,
            std::vector<ValueType> &rho_flat,
            std::vector<uint64_t> &pi,
            uint64_t numObjs) :
            transitionMatrix_(transitionMatrix), rho_(rho_flat), pi_(pi),
            stateIndices_(rowGroupIndices), numObjs_(numObjs) {}

    template<typename ValueType>
    int CudaIVHandler<ValueType>::initialise(){

        A_nnz = transitionMatrix_.nonZeros();
        A_ncols = transitionMatrix_.cols();
        A_nrows = transitionMatrix_.rows();
        //std::vector<ValueType> x(A_ncols, static_cast<ValueType>(0.0));
        //std::vector<ValueType> y(A_nrows, static_cast<ValueType>(0.0));
        printf("____GOT HERE!!____\n");
        CHECK_CUDA(cudaMalloc((void **) &dA_csrOffsets, (transitionMatrix_.rows() + 1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dA_values, A_nnz * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dXPrime, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dXTemp, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dR, rho_.size() * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dEnabledActions, A_ncols * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dPi, A_ncols * sizeof(uint)))
        printf("____GOT HERE!!____\n");
        CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix_.outerIndexPtr(),
                              (A_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix_.innerIndexPtr(),
                              A_nnz * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix_.valuePtr(),
                              A_nnz * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemcpy(dX, x.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemset(dX, static_cast<double>(0.0), A_ncols * sizeof(double)))
        //CHECK_CUDA(cudaMemcpy(dXPrime, x.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemset(dXPrime, static_cast<double>(0.0), A_ncols * sizeof(double)))
        //CHECK_CUDA(cudaMemcpy(dY, y.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemset(dY, static_cast<double>(0.0), A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMemcpy(dR, rho_.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice))
        //CHECK_CUDA(cudaMemcpy(dXTemp, x.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemset(dXTemp, static_cast<double>(0.0), A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMemcpy(dEnabledActions, stateIndices_.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dPi, pi_.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice))
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
        // Create dense vector R
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, A_nrows, dR, CUDA_R_64F))
        // allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
        printf("____GOT HERE!!____\n");

        return EXIT_SUCCESS;
    }


    template<typename ValueType>
    int CudaIVHandler<ValueType>::valueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> const &transitionMatrix,
                                             std::vector<ValueType> &x,
                                             std::vector<ValueType> &r,
                                             std::vector<int> &pi,
                                             std::vector<int> const &rowGroupIndices) {

        //-------------------------------------------------------------------------
        // Device memory management
        int *dA_csrOffsets, *dA_columns, *dEnabledActions, *dPi;
        double *dA_values, *dX, *dY, *dR, *dXTemp, *dXPrime;
        int A_nnz, A_ncols, A_nrows;
        A_nnz = transitionMatrix.nonZeros();
        A_ncols = transitionMatrix.cols();
        A_nrows = transitionMatrix.rows();
        double alpha = 1.0;
        double beta = 1.0;
        double eps = 1.0;

        std::vector<ValueType> y(r.size(), static_cast<ValueType>(0.0));

        CHECK_CUDA(cudaMalloc((void **) &dA_csrOffsets, (transitionMatrix.rows() + 1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dA_values, A_nnz * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dX, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dXPrime, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dXTemp, A_ncols * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dY, A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dR, A_nrows * sizeof(double)))
        CHECK_CUDA(cudaMalloc((void **) &dEnabledActions, A_ncols * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void **) &dPi, A_ncols * sizeof(uint)))


        CHECK_CUDA(cudaMemcpy(dA_csrOffsets, transitionMatrix.outerIndexPtr(),
                              (A_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dA_columns, transitionMatrix.innerIndexPtr(),
                              A_nnz * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_values, transitionMatrix.valuePtr(),
                              A_nnz * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dX, x.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dXPrime, x.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dY, y.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dR, r.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dXTemp, x.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dEnabledActions, rowGroupIndices.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dPi, pi.data(), A_ncols * sizeof(int), cudaMemcpyHostToDevice))
        //-------------------------------------------------------------------------
        //CUSPARSE APIs
        cublasHandle_t cublasHandle = nullptr;
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY, vecR;
        void *dBuffer = nullptr;
        size_t bufferSize = 0;

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
        // Create dense vector R
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, A_nrows, dR, CUDA_R_64F))
        // allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

        //-------------------------------------------------------------------------
        // Execute value iteration
        double maxEps = 0.0, max = 1.;
        int iterations = 0;
        double alpha2 = -1.0;
        do {
            // y = r
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_nrows, dR, 1, dY, 1))
            // y = A.x + r(y = r)
            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
            // compute the next policy update
            // This updates x with the maximum value associated with an action
            /*
            CHECK_CUDA(cudaMemcpy(y.data(), dY, A_nrows * sizeof(double), cudaMemcpyDeviceToHost))
            nextBestPolicy(y, x, pi, rowGroupIndices);
            CHECK_CUDA(cudaMemcpy(dX, x.data(), A_ncols * sizeof(double), cudaMemcpyHostToDevice))
            CHECK_CUDA(cudaMemcpy(dY, y.data(), A_nrows * sizeof(double), cudaMemcpyHostToDevice))*/
            mopmc::kernels::maxValueLauncher(dY, dX, dEnabledActions, dPi, A_ncols);
            // In the following computation dXTemp will be changed to dXTemp = dX - dXTemp
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXPrime, 1))

            /*std::cout << "X: \n";
            CHECK_CUDA(cudaMemcpy(x.data(), dX, A_ncols * sizeof(double), cudaMemcpyDeviceToHost))
            for (int i = 0 ; i < 10 ; ++ i) {
                std::cout << x[i] << " ";
            }
            std::cout << "\n";
            */
            CHECK_CUBLAS(cublasDaxpy_v2_64(cublasHandle, A_ncols, &alpha2, dXTemp, 1, dXPrime, 1))

            maxEps = mopmc::kernels::findMaxEps(dXPrime, A_ncols, maxEps);
            //CHECK_CUDA(cudaMemcpy(&max, maxEps, sizeof(double), cudaMemcpyDeviceToHost))
            CHECK_CUBLAS(cublasDcopy_v2_64(cublasHandle, A_ncols, dX, 1, dXTemp, 1))
            //
            //mopmc::kernels::launchPrintKernel(maxEps);

            // Copy the dX values over to dXTemp
            /*if (iterations > 10) {
                break;
            }*/
            ++iterations;
        } while (maxEps > 1e-5);

        CHECK_CUDA(cudaMemcpy(x.data(), dX, A_ncols * sizeof(double), cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(pi.data(), dPi, A_ncols * sizeof(int), cudaMemcpyDeviceToHost))
        CHECK_CUDA(cudaMemcpy(r.data(), dR, A_nrows * sizeof(double), cudaMemcpyDeviceToHost))

        //std::cout << "max Eps: " << maxEps <<"\n";


        //-------------------------------------------------------------------------
        // destroy matrix/vector descriptors
        CHECK_CUSPARSE(cusparseDestroySpMat(matA))
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
        CHECK_CUSPARSE(cusparseDestroy(handle))

        // device memory deallocation
        CHECK_CUDA(cudaFree(dBuffer))
        CHECK_CUDA(cudaFree(dA_csrOffsets))
        CHECK_CUDA(cudaFree(dA_columns))
        CHECK_CUDA(cudaFree(dA_values))
        CHECK_CUDA(cudaFree(dX))
        CHECK_CUDA(cudaFree(dY))

        return EXIT_SUCCESS;
    }


    /*
template int
CudaIVHandler<double>::valueIteration(Eigen::SparseMatrix<double, Eigen::RowMajor> const &transitionMatrix,
                              std::vector<double> &x,
                              std::vector<double> &r,
                              std::vector<int> &pi,
                              std::vector<int> const &rowGroupIndices);

template int CudaIVHandler<double>::initialise();
 */

    template class CudaIVHandler<double>;
}