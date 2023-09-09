//
// Created by thomas on 7/09/23.
//

#ifndef MOPMC_ITERATIVESOLVER_H
#define MOPMC_ITERATIVESOLVER_H

#include <storm/utility/constants.h>
#include <Eigen/Sparse>
#include <storm/storage/SparseMatrix.h>

namespace mopmc {
namespace solver::iter{

template <typename ValueType>
void valueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                    Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                    std::vector<ValueType> &r, std::vector<uint64_t> &pi,
                    std::vector<typename storm::storage::SparseMatrix<ValueType>::index_type> const& rowGroupIndices);


template <typename ValueType>
bool nextBestPolicy(Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &y,
                    Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                    std::vector<uint64_t> &pi,
                    std::vector<typename storm::storage::SparseMatrix<ValueType>::index_type> const& rowGroupIndices);

template <typename ValueType>
ValueType computeEpsilon(Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &xprev,
                    Eigen::Index &maxRow);

template <typename SparseModelType, typename ValueType>
void policyIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                     Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &identity,
                     Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                     std::vector<ValueType> &r, std::vector<uint64_t> &pi,
                     std::vector<typename storm::storage::SparseMatrix<ValueType>::index_type> const& rowGroupIndices);


}
}


#endif //MOPMC_ITERATIVESOLVER_H
