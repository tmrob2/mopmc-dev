//
// Created by thomas on 12/08/23.
//

#ifndef MOPMC_MODELCHECKINGUTILS_H
#define MOPMC_MODELCHECKINGUTILS_H

#include <storm/storage/BitVector.h>
#include "../SparseModel.h"
#include <Eigen/Sparse>

namespace mopmc {
namespace sparseutils{
template<typename T>
storm::storage::BitVector findStatesProbGreater0(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates,
    bool useStepBound = false,
    uint_fast64_t maximalSteps = 0
);

template <typename T>
storm::storage::BitVector findStatesProbEq1(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& psiStates,
    storm::storage::BitVector const& statesWithProbGe0
);

template <typename T>
std::pair<storm::storage::BitVector, storm::storage::BitVector> performProb01(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
);

template <typename T>
storm::storage::BitVector getOneStep(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& beackwardTransitions,
    storm::storage::BitVector const& psiStates
);

} // sparseutils
} // mopmc


#endif //MOPMC_MODELCHECKINGUTILS_H
