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


template <typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> eigenInducedTransitionMatrix(
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> &fullTransitionSystem,
    std::vector<uint64_t>& scheduler,
    std::vector<uint_fast64_t> const& rowGroupIndices);

template <typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> makeIdentity(uint_fast64_t n);

template<typename ValueType>
void inducedRewards(std::vector<ValueType> &b, std::vector<ValueType> &subB,
                    std::vector<uint64_t> &scheduler, std::vector<uint_fast64_t> const &rowGroupIndices);

template<typename SparseModelType>
storm::storage::BitVector performProbGreater0(
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> const& backwardTransitions,
        storm::storage::BitVector const& phiStates,
        storm::storage::BitVector const& psiStates);

template<typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>
constructSubMatrix(Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> &dtmc,
                   storm::storage::BitVector &maybeStates,
                   std::unordered_map<uint_fast64_t, uint_fast64_t>& map);

template<typename SparseModelType>
void makeRhs(std::vector<typename SparseModelType::ValueType>& b,
             std::vector<typename SparseModelType::ValueType>const& deterministicStateRewards,
             //std::vector<uint_fast64_t>const& rowGroupIndices,
             storm::storage::BitVector& maybeStates);
             //std::vector<uint_fast64_t>& scheduler);

}

#endif //MOPMC_SOLVERHELPER_H
