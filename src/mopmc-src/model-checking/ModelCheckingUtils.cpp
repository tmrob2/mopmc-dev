//
// Created by thomas on 12/08/23.
//
#include <storm/storage/BitVector.h>
#include "../SparseModel.h"
#include <Eigen/Sparse>
#include "ModelCheckingUtils.h"

template<typename T>
storm::storage::BitVector mopmc::sparseutils::findStatesProbGreater0(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates,
    bool useStepBound,
    uint_fast64_t maximalSteps
){
    // prepare the resulting bitvector
    uint_fast64_t numberOfStates = phiStates.size();
    storm::storage::BitVector statesWithProbabilityGreater0(numberOfStates);

    // Add all psi states as they already satisfy the condition
    statesWithProbabilityGreater0 |= psiStates;

    std::cout << std::endl;

    // Initialise the stack used for the DFS with the starting states
    std::cout << "psiStates: ";
    for (T val : psiStates) {
        std::cout << val << ", ";
    }
    std::cout << "\n";
    std::vector<uint_fast64_t> stack(psiStates.begin(), psiStates.end());

    std::vector<uint_fast64_t> stepStack;
    std::vector<uint_fast64_t> remainingSteps;
    if (useStepBound) {
        stepStack.reserve(numberOfStates);
        stepStack.insert(stepStack.begin(), psiStates.getNumberOfSetBits(), maximalSteps);
        remainingSteps.resize(numberOfStates);
        for(auto state : psiStates) {
            remainingSteps[state] = maximalSteps;
        }
    }

    std::cout << backwardTransitions.toDense() << std::endl;

    // Perform the DFS
    uint_fast64_t currentState, currentStepBound;
    while(!stack.empty()) {
        currentState = stack.back();
        stack.pop_back();

        if (useStepBound) {
            currentStepBound = stepStack.back();
            stepStack.pop_back();
            if (currentStepBound == 0) {
                continue;
            }
        }

        /* OK so for this next bit we are going to hack out the storm code a bit.
         * We need a row from the sparse matrix which we can iterator over.
         * Select a row from the matrix
        */
        typename mopmc::sparse::SparseModelBuilder<T>::SpMat::InnerIterator it(backwardTransitions, currentState);
        for(; it; ++it) {
            if (phiStates[it.col()] &&
                (!statesWithProbabilityGreater0.get(it.col()) ||
                 (useStepBound && remainingSteps[it.col()] < currentStepBound - 1))) {
                statesWithProbabilityGreater0.set(it.col(), true);
                stack.push_back(it.col());
            }
        }
    }
    return statesWithProbabilityGreater0;
}

template <typename T>
storm::storage::BitVector mopmc::sparseutils::findStatesProbEq1(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& psiStates,
    storm::storage::BitVector const& statesWithProbGe0
) {
    storm::storage::BitVector statesWithProb1 =
            mopmc::sparseutils::findStatesProbGreater0<T>(backwardTransitions, ~psiStates, ~statesWithProbGe0);
    statesWithProb1.complement();
    return statesWithProb1;
}

template <typename T>
std::pair<storm::storage::BitVector, storm::storage::BitVector> mopmc::sparseutils::performProb01(
        typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
        storm::storage::BitVector const& phiStates,
        storm::storage::BitVector const& psiStates
){
    std::pair<storm::storage::BitVector, storm::storage::BitVector> result;
    result.first = mopmc::sparseutils::findStatesProbGreater0<T>(backwardTransitions, phiStates, psiStates);
    result.second = mopmc::sparseutils::findStatesProbEq1<T>(backwardTransitions,psiStates, result.first);
    result.first.complement(); // The idea here is to return states with Prob 0 i.e. not (Pr > 0)
    return result;
};

template <typename T>
storm::storage::BitVector mopmc::sparseutils::getOneStep(
    typename mopmc::sparse::SparseModelBuilder<T>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& psiStates
) {
    storm::storage::BitVector oneStepStates(psiStates.size(), false);
    for (uint_fast64_t i = 0; i < backwardTransitions.rows(); ++i){
        uint_fast64_t rowStart = backwardTransitions.outerIndexPtr()[i];
        uint_fast64_t rowEnd = backwardTransitions.outerIndexPtr()[i+1];
        if (psiStates[i]) {
            for (uint_fast64_t k = rowStart; k < rowEnd; ++k) {
                uint_fast64_t colIndex = backwardTransitions.innerIndexPtr()[k];
                oneStepStates.set(colIndex);
            }
        }
    }

    oneStepStates = oneStepStates ^ psiStates; // need to make sure we don't include
    
    return oneStepStates;
}

template <typename T>
void mopmc::sparseutils::setVector(std::vector<T>& vectorToUpdate, storm::storage::BitVector const& updatePositions,
               Eigen::Matrix<T, Eigen::Dynamic, 1> solverResult){
    uint_fast64_t oldPosition = 0;
    for(uint_fast64_t pos: updatePositions) {
        vectorToUpdate[pos] = solverResult[oldPosition++];
    }
}

// Explicit Instantiations
template storm::storage::BitVector mopmc::sparseutils::findStatesProbEq1<double>(
    typename mopmc::sparse::SparseModelBuilder<double>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& psiStates,
    storm::storage::BitVector const& statesWithProbGe0
);

template storm::storage::BitVector mopmc::sparseutils::findStatesProbGreater0<double>(
    typename mopmc::sparse::SparseModelBuilder<double>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates,
    bool useStepBound,
    uint_fast64_t maximalSteps
);

template std::pair<storm::storage::BitVector, storm::storage::BitVector> mopmc::sparseutils::performProb01<double>(
    typename mopmc::sparse::SparseModelBuilder<double>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
);

template storm::storage::BitVector mopmc::sparseutils::getOneStep<double>(
    typename mopmc::sparse::SparseModelBuilder<double>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& psiStates
);

template void mopmc::sparseutils::setVector<double>(
    std::vector<double> &vectorToUpdate,const storm::storage::BitVector &updatePositions,
    Eigen::Matrix<double, Eigen::Dynamic, 1> solverResult);


