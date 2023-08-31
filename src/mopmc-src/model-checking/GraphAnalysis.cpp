//
// Created by thomas on 12/08/23.
//
#include <storm/storage/BitVector.h>
#include "../SparseModel.h"
#include <Eigen/Sparse>
#include "GraphAnalysis.h"

template<typename T>
storm::storage::BitVector mopmc::graph::findStatesProbGreater0(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates,
    bool useStepBound,
    uint_fast64_t maximalSteps
){
    // prepare the resulting bitvector
    uint_fast64_t numberOfStates = phiStates.size();
    auto backwardTransitions = model.getBackwardTransitions();
    auto& reverseStateActionMapping = model.getReverseStateActionMapping();
    storm::storage::BitVector statesWithProbabilityGreater0(numberOfStates);

    // Add all psi states as they already satisfy the condition
    statesWithProbabilityGreater0 |= psiStates;
    std::vector<uint_fast64_t> stack(psiStates.begin(), psiStates.end());

    std::vector<uint_fast64_t> stepStack;
    std::vector<uint_fast64_t> remainingSteps;
    if (useStepBound) {
        throw std::runtime_error("Step bounded properties are not implemented: Graph Analaysis:findStatesProbGreater0");
        /*stepStack.reserve(numberOfStates);
        stepStack.insert(stepStack.begin(), psiStates.getNumberOfSetBits(), maximalSteps);
        remainingSteps.resize(numberOfStates);
        for(auto state : psiStates) {
            remainingSteps[state] = maximalSteps;
        }*/
    }

    //std::cout << backwardTransitions.toDense() << std::endl;

    // Perform the DFS
    uint_fast64_t currentState, currentStepBound;
    while(!stack.empty()) {
        currentState = stack.back();
        stack.pop_back();

        /* OK so for this next bit we are going to hack the storm code a bit.
         * We need a row from the sparse matrix which we can iterator over.
         * Select a row from the matrix
         *
         * A mapping is used because we have to deal with non-determinism
        */
        typename mopmc::sparse::SparseModelBuilder<T>::SpMat::InnerIterator it(backwardTransitions, currentState);
        for(; it; ++it) {
            if (phiStates[reverseStateActionMapping[it.col()]] &&
                (!statesWithProbabilityGreater0.get(reverseStateActionMapping[it.col()]))) {
                statesWithProbabilityGreater0.set(reverseStateActionMapping[it.col()], true);
                stack.push_back(reverseStateActionMapping[it.col()]);
            }
        }
    }
    return statesWithProbabilityGreater0;
}

template <typename T>
storm::storage::BitVector mopmc::graph::performProb1A(
    typename mopmc::sparse::SparseModelBuilder<T> &spModel,
    const storm::storage::BitVector &phiStates,
    const storm::storage::BitVector &psiStates) {

    typename mopmc::sparse::SparseModelBuilder<T>::SpMat backwardTransitions =
            spModel.getBackwardTransitions();

    size_t numberOfStates = phiStates.size();

    storm::storage::BitVector currentStates(numberOfStates, true);
    std::vector<uint_fast64_t> stack;
    stack.reserve(numberOfStates);

    bool done = false;
    // perform the loop as long as the set of states gets smaller
    uint_fast64_t currentState;
    std::unordered_map<uint_fast64_t, uint_fast64_t> reverseMap =
            spModel.getReverseStateActionMapping();
    while(!done){
        stack.clear();
        storm::storage::BitVector nextStates(psiStates);
        stack.insert(stack.end(), psiStates.begin(), psiStates.end());

        while(!stack.empty()) {

            currentState = stack.back();
            stack.pop_back();

            // loop over all the non-empty predecessors.
            for (uint_fast64_t k = 0; k < backwardTransitions.outerSize(); ++k) {
                for (typename mopmc::sparse::SparseModelBuilder<T>::SpMat::InnerIterator
                    it(backwardTransitions, k); it; ++it){
                    if (phiStates.get(reverseMap[it.col()]) && !nextStates.get(reverseMap[it.col()])) {
                        // check whether the predecessor has only successors in the current state
                        // set for all of the non-deterministic choices and that for each choice there
                        // exists a successor that is already in the next states
                        bool addToStatesWithProbability1 = true;
                        for(uint_fast64_t row : spModel.getActionsForState(reverseMap[it.col()])) {
                            bool hasAtLeastOneSuccessorWithProbability1 = false;
                            for(uint_fast64_t k2 = 0; k2 < spModel.getTransitionMatrix().outerSize(); ++k2) {
                                for (typename mopmc::sparse::SparseModelBuilder<T>::SpMat::InnerIterator
                                    itForward(spModel.getTransitionMatrix(), k2); itForward; ++itForward){
                                    if (!currentStates.get(itForward.col())) {
                                        addToStatesWithProbability1 = false;
                                        goto afterCheckLoop;
                                    }

                                    if (nextStates.get(itForward.col())){
                                        hasAtLeastOneSuccessorWithProbability1 = true;
                                    }
                                }
                            }
                            if(!hasAtLeastOneSuccessorWithProbability1) {
                                addToStatesWithProbability1 = false;
                                break;
                            }
                        }
                    afterCheckLoop:
                        // If all successors for all nondeterministic choices are in the current
                        // state set we add tit to the set of states for the next iteration and
                        // perform a backward search from that state
                        if(addToStatesWithProbability1) {
                            nextStates.set(reverseMap[it.col()], true);
                            stack.push_back(reverseMap[it.col()]);
                        }

                    }
                }
            }
        }

        // Cehck whether we need to perform an additional iteration
        if(currentStates == nextStates) {
            done = true;
        } else {
            std::cout<<"Current states does not eual next States\n";
            currentStates = std::move(nextStates);
        }

    }
    return currentStates;
}


template<typename T>
storm::storage::BitVector mopmc::graph::performProb0A(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
) {
    storm::storage::BitVector statesWithProbability0 = mopmc::graph::findStatesProbGreater0<T>(
        model, phiStates, psiStates);
    statesWithProbability0.complement();
    return statesWithProbability0;
}

template <typename T>
storm::storage::BitVector mopmc::graph::findStatesProbEq1(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& psiStates,
    storm::storage::BitVector const& statesWithProbGe0
) {
    storm::storage::BitVector statesWithProb1 =mopmc::graph::findStatesProbGreater0<T>(
                    model, ~psiStates, ~statesWithProbGe0);
    statesWithProb1.complement();
    return statesWithProb1;
}

template <typename T>
std::pair<storm::storage::BitVector, storm::storage::BitVector> mopmc::graph::performProb01(
    typename mopmc::sparse::SparseModelBuilder<T>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
){
    std::pair<storm::storage::BitVector, storm::storage::BitVector> result;
    result.first = mopmc::graph::findStatesProbGreater0<T>(
            model, phiStates, psiStates);
    result.second = mopmc::graph::findStatesProbEq1<T>(
            model,psiStates,result.first);
    result.first.complement(); // The idea here is to return states with Prob 0 i.e. not (Pr > 0)
    return result;
};

template <typename T>
storm::storage::BitVector mopmc::graph::getOneStep(
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
void mopmc::graph::setVector(std::vector<T>& vectorToUpdate, storm::storage::BitVector const& updatePositions,
                             Eigen::Matrix<T, Eigen::Dynamic, 1> solverResult){
    uint_fast64_t oldPosition = 0;
    for(uint_fast64_t pos: updatePositions) {
        vectorToUpdate[pos] = solverResult[oldPosition++];
    }
}

template <typename T>
storm::storage::BitVector mopmc::graph::getReachableStates(
        typename mopmc::sparse::SparseModelBuilder<T>& model,
        storm::storage::BitVector const& initialStates,
        storm::storage::BitVector const& constraintStates,
        storm::storage::BitVector const& targetStates,
        boost::optional<storm::storage::BitVector> const& choiceFilter
) {
    storm::storage::BitVector reachableStates(initialStates);
    auto& transitionMatrix = (sparse::SparseModelBuilder<double>::SpMat &) model.getTransitionMatrix();
    uint_fast64_t numberOfStates = transitionMatrix.cols();

    // initialise the stack for DFS
    std::vector<uint_fast64_t> stack;
    stack.reserve(initialStates.size()); // in reality because we only deal with multi-objective this will only be one
    if (choiceFilter) {
        std::cout << "Initial states size: " << initialStates.getNumberOfSetBits() << "\n";
        std::cout << "Constraint set size: " << constraintStates.getNumberOfSetBits() << "\n";
    }
    for(uint_fast64_t state: initialStates) {
        std::cout << "initial state: " << state << "\n";
        if(constraintStates.get(state)) {
            stack.push_back(state);
        } else {
            std::cout << "initial state not constraint state\n";
        }
    }

    // step bounded properties are also illegal
    // perform the actual DFS
    uint_fast64_t currentState = 0;
    while (!stack.empty()) {
        currentState = stack.back();
        stack.pop_back();
        // only loop over rows that contain matrix entries
        std::vector<uint_fast64_t>& actions = model.getActionsForState(currentState);
        std::vector<uint_fast64_t> actionsCp(actions.size());
        // using the choice filter we only reach those states from which there are enabled actions
        if (choiceFilter) {
            for(uint_fast64_t action: actions) {
                if (choiceFilter->get(action)) {
                    actionsCp.push_back(action);
                }
            }
        } else {
            actionsCp = actions;
        }
        for (uint_fast64_t action : actionsCp) {
            typename mopmc::sparse::SparseModelBuilder<T>::SpMat::InnerIterator it(transitionMatrix, action);
            for(; it; ++it) {
                if (!reachableStates.get(it.col())) {
                    // If the success is one of the target states we inlcude it but don't explore
                    // further
                    if (targetStates.get(it.col())){
                        reachableStates.set(it.col());
                    } else if (constraintStates.get(it.col())){
                        // potentially follow this state further
                        reachableStates.set(it.col());
                        stack.push_back(it.col());
                    }
                }
            }
        }
    }
    std::cout << "Reachable states: " << reachableStates.getNumberOfSetBits() << "\n";
    return reachableStates;
}

//#####################################################
// Explicit Instantiations
//#####################################################
template storm::storage::BitVector mopmc::graph::findStatesProbEq1<double>(
    typename mopmc::sparse::SparseModelBuilder<double>& model,
    storm::storage::BitVector const& psiStates,
    storm::storage::BitVector const& statesWithProbGe0
);

template storm::storage::BitVector mopmc::graph::findStatesProbGreater0<double>(
    typename mopmc::sparse::SparseModelBuilder<double> &model,
    const storm::storage::BitVector &phiStates,
    const storm::storage::BitVector &psiStates,
    bool useStepBound,
    uint_fast64_t maximalSteps
);

template std::pair<storm::storage::BitVector, storm::storage::BitVector> mopmc::graph::performProb01<double>(
    typename mopmc::sparse::SparseModelBuilder<double>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
);

template storm::storage::BitVector mopmc::graph::performProb0A<double>(
    typename mopmc::sparse::SparseModelBuilder<double>& model,
    storm::storage::BitVector const& phiStates,
    storm::storage::BitVector const& psiStates
);

template storm::storage::BitVector mopmc::graph::performProb1A<double>(
        typename mopmc::sparse::SparseModelBuilder<double> &spModel,
        const storm::storage::BitVector &phiStates,
        const storm::storage::BitVector &psiStates);

template storm::storage::BitVector mopmc::graph::getOneStep<double>(
    typename mopmc::sparse::SparseModelBuilder<double>::SpMat const& backwardTransitions,
    storm::storage::BitVector const& psiStates
);

template void mopmc::graph::setVector<double>(
    std::vector<double> &vectorToUpdate,const storm::storage::BitVector &updatePositions,
    Eigen::Matrix<double, Eigen::Dynamic, 1> solverResult);

template storm::storage::BitVector mopmc::graph::getReachableStates<double>(
    typename mopmc::sparse::SparseModelBuilder<double>& model,
    storm::storage::BitVector const& initialStates, storm::storage::BitVector const& constraintStates,
    storm::storage::BitVector const& targetStates,
    boost::optional<storm::storage::BitVector> const& choiceFilter
);
