//
// Created by thomas on 12/08/23.
//

#ifndef MOPMC_GRAPHANALYSIS_H
#define MOPMC_GRAPHANALYSIS_H

#include <storm/storage/BitVector.h>
#include "SparseModel.h"
#include <Eigen/Sparse>
#include <boost/optional/optional.hpp>

namespace mopmc {
namespace graph{

/*!
 * Handles both (non-)deterministic models
 * @tparam T: Value Type generic
 */
template<typename T>
storm::storage::BitVector findStatesProbGreater0(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates,
    bool useStepBound = false,
    uint_fast64_t maximalSteps = 0
);

template <typename T>
storm::storage::BitVector findStatesProbEq1(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& psiStates,
    storm::storage::BitVector const& statesWithProbGe0
);

template <typename T>
std::pair<storm::storage::BitVector, storm::storage::BitVector> performProb01(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
);

template<typename T>
storm::storage::BitVector performProb0A(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
);

template<typename T>
storm::storage::BitVector performProb1A(
    typename mopmc::sparse::SparseModelBuilder<T>& spModel,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
);

template <typename T>
storm::storage::BitVector getOneStep(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& psiStates
);

template <typename T>
storm::storage::BitVector getReachableStates(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& initialStates, storm::storage::BitVector const& constraintStates,
    storm::storage::BitVector const& targetStates,
    boost::optional<storm::storage::BitVector> const& choiceFilter
);

template <typename T>
void setVector(std::vector<T>& vectorToUpdate, storm::storage::BitVector const& updatePositions,
               Eigen::Matrix<T, Eigen::Dynamic, 1>);

} // graph
} // mopmc


#endif //MOPMC_GRAPHANALYSIS_H
