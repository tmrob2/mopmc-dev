//
// Created by thomas on 26/09/23.
//

#ifndef MOPMC_CUVISOLVER_H
#define MOPMC_CUVISOLVER_H

#include <storm/utility/constants.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <Eigen/Sparse>
#include <storm/storage/SparseMatrix.h>

namespace mopmc::solver::cuda{

template <typename ValueType>
int valueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> const& transitionSystem,
                    std::vector<ValueType>& x,
                    std::vector<ValueType>& r,
                    std::vector<int>& pi,
                    std::vector<int> const& rowGroupIndices);
}

#endif //MOPMC_CUVISOLVER_H
