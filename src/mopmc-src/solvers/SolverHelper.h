//
// Created by thomas on 26/09/23.
//

#ifndef MOPMC_SOLVERHELPER_H
#define MOPMC_SOLVERHELPER_H

#include <storm/utility/constants.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <storm/storage/SparseMatrix.h>

namespace mopmc::solver::helper {
template <typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> eigenInducedTransitionMatrix(
    storm::storage::SparseMatrix<typename SparseModelType::ValueType> &fullTransitionSystem,
    //Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> &b,
    //Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> &subB,
    std::vector<uint64_t>& scheduler,
    std::vector<uint_fast64_t> const& rowGroupIndices);

template<typename ValueType>
void inducedRewards(std::vector<ValueType> &b, std::vector<ValueType> &subB,
                    std::vector<uint64_t> &scheduler, std::vector<uint_fast64_t> const &rowGroupIndices);
}

#endif //MOPMC_SOLVERHELPER_H
