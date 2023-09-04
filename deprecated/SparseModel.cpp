//
// Created by thomas on 8/08/23.
//

#include "SparseModel.h"
#include "GraphAnalysis.h"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <iostream>

namespace mopmc::sparse{
    template<typename ValueType>
    typename sparse::SparseModelBuilder<ValueType>::SpMat& SparseModelBuilder<ValueType>::getSparseMatrix() {
        return this -> transitionMatrix;
    }

    template <typename ValueType>
    uint_fast64_t& SparseModelBuilder<ValueType>::getNumberOfStates() {
        return this -> numberOfStates;
    }

    template <typename ValueType>
    void SparseModelBuilder<ValueType>::setNumberOfStates(uint_fast64_t num_states) {
        this -> numberOfStates = num_states;
    }

    template <typename ValueType>
    uint_fast64_t& SparseModelBuilder<ValueType>::getNumberOfChoices() {
        return this -> numberOfChoices;
    }

    template <typename ValueType>
    void SparseModelBuilder<ValueType>::setNumberOfChoices(uint_fast64_t numChoices) {
        this -> numberOfChoices = numChoices;
    }

    template <typename ValueType>
    uint_fast64_t SparseModelBuilder<ValueType>::getNumberOfTransitions() {
        return this -> numberOfTransitions;
    }

    template <typename ValueType>
    void SparseModelBuilder<ValueType>::setNumberOfTransitions(uint_fast64_t numTransitions){
        this -> numberOfTransitions = numTransitions;
    }

    template <typename ValueType>
    void SparseModelBuilder<ValueType>::setStateLabels(const storm::models::sparse::StateLabeling& sLabels){
        this -> stateLabels = sLabels;
    };

    template <typename ValueType>
    storm::storage::BitVector const& SparseModelBuilder<ValueType>::getStates(std::string const& label) const {
        return this->stateLabels.getStates(label);
    };

    template <typename ValueType>
    bool SparseModelBuilder<ValueType>::hasLabel(std::string const& label){
        return stateLabels.containsLabel(label);
    }

    template <typename ValueType>
    typename SparseModelBuilder<ValueType>::SpMat SparseModelBuilder<ValueType>::getBackwardTransitions() {
        return transitionMatrix.transpose();
    }

    template <typename ValueType>
    typename SparseModelBuilder<ValueType>::SpMat const& SparseModelBuilder<ValueType>::getTransitionMatrix() {
        return this -> transitionMatrix;
    }

    template <typename ValueType>
    const storm::storage::BitVector& SparseModelBuilder<ValueType>::getInitialStates() const {
        return this -> getStates("init");
    }

    template <typename ValueType>
    std::unordered_map<uint_fast64_t, std::vector<uint_fast64_t>> const& SparseModelBuilder<ValueType>::getStateActionMapping() {
        return this -> stateActionMapping;
    }

    template <typename ValueType>
    void SparseModelBuilder<ValueType>::addNewActionToState(uint_fast64_t state, uint_fast64_t action_index) {
        auto it = this -> stateActionMapping.find(state);
        if (it != stateActionMapping.end()) {
            it->second.push_back(action_index);
        } else {
            this->stateActionMapping[state] = {action_index};
        }
    };

    template<typename ValueType>
    const std::vector<uint_fast64_t> & SparseModelBuilder<ValueType>::getNumberActionsForState(uint_fast64_t state) {
        return this -> stateActionMapping[state];
    }

    template<typename ValueType>
    std::unordered_map<uint_fast64_t, uint_fast64_t>& SparseModelBuilder<ValueType>::getReverseStateActionMapping() {
        return this -> reverseStateActionMapping;
    };

    template <typename ValueType>
    void SparseModelBuilder<ValueType>::insertReverseStateActionMap(uint_fast64_t state, uint_fast64_t actionIndex) {
        reverseStateActionMapping[actionIndex] = state;
    };

    template <typename ValueType>
    std::vector<uint_fast64_t>& SparseModelBuilder<ValueType>::getActionsForState(uint_fast64_t state) {
        return this->stateActionMapping[state];
    }

    template<typename ValueType>
    void SparseModelBuilder<ValueType>::insertRewardModel(
        std::string rewardModelName,
        storm::models::sparse::StandardRewardModel<ValueType> rewardModel) {
        this->rewardModels.emplace(rewardModelName, rewardModel);
    }

    template<typename ValueType>
    storm::models::sparse::StandardRewardModel<ValueType>& SparseModelBuilder<ValueType>::getRewardModel(
            std::string rewardModelName) {
        return this->rewardModels[rewardModelName];
    }

    template <typename ValueType>
    std::vector<std::string> SparseModelBuilder<ValueType>::getRewardModelNames(){
        std::vector<std::string> keys;
        for(auto it = this->rewardModels.begin(); it != this->rewardModels.end(); ++it){
            const std::string& key = it->first;
            keys.push_back(key);
        }
        return keys;
    }

    template <typename ValueType>
    std::pair<typename SparseModelBuilder<ValueType>::SpMat, std::unordered_map<uint_fast64_t, uint_fast64_t>>
            SparseModelBuilder<ValueType>::getDTMCSubMatrix(
        const storm::storage::BitVector &maybeStates) {

        uint_fast64_t subMatrixSize = maybeStates.getNumberOfSetBits();
        std::cout << "Sub matrix size: " << subMatrixSize << "\n";

        // create a hashmap which stores the state to its compressed value
        std::unordered_map<uint_fast64_t, uint_fast64_t> subMatMap;
        uint_fast64_t newIndex = 0;
        for(int i = 0; i != maybeStates.size() ; ++i) {
            if (maybeStates[i]) {
                subMatMap.emplace(i, newIndex);
                ++newIndex;
            }
        }
        SpMat subMatrix(subMatrixSize, subMatrixSize);

        for (int k = 0; k < transitionMatrix.outerSize(); ++k) {
            for(typename SpMat::InnerIterator it(transitionMatrix, k); it; ++it) {
                if(maybeStates[it.row()] && maybeStates[it.col()]) {
                    subMatrix.insert(subMatMap[it.row()], subMatMap[it.col()]) = it.value();
                }
            }
        }
        subMatrix.makeCompressed();
        return std::make_pair(subMatrix, subMatMap);
    }

    template<typename ValueType>
    void SparseModelBuilder<ValueType>::getMDPSubMatrix(
        const storm::storage::BitVector &subsystemStates,
        const storm::storage::BitVector &subsystemActions) {

        uint_fast64_t matrixRows = subsystemActions.getNumberOfSetBits();
        uint_fast64_t matrixCols = subsystemStates.getNumberOfSetBits();

        std::unordered_map<uint_fast64_t, uint_fast64_t> subStateMap;
        std::unordered_map<uint_fast64_t, uint_fast64_t> subActionMap;
        uint_fast64_t newStateIndex = 0;
        uint_fast64_t newActionIndex = 0;
        for(uint_fast64_t state: subsystemStates) {
            subStateMap.emplace(state, newStateIndex);
            newStateIndex++;
        }
        for(uint_fast64_t action: subsystemActions) {
            subActionMap.emplace(action, newActionIndex);
            newActionIndex++;
        }
        SpMat subMatrix(matrixRows, matrixCols);
        for(uint_fast64_t state: subsystemStates) {
            std::vector<uint_fast64_t> const& actions = this->getActionsForState(state);
            for(uint_fast64_t action : actions ) {
               if(subsystemActions[action]) {
                   typename SpMat::InnerIterator it(this->transitionMatrix, action);
                   for(; it; ++it){
                       subMatrix.insert(subActionMap[it.row()],subStateMap[it.col()]) = it.value();
                   }
               }
            }
        }
        subMatrix.makeCompressed();
        this -> transitionMatrix = subMatrix;
    }

    template <typename ValueType>
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> SparseModelBuilder<ValueType>::bVector(
            storm::storage::BitVector const& prob1States,
            SpMat const& backwardTransitions,
            uint_fast64_t dim,
            std::unordered_map<uint_fast64_t, uint_fast64_t>& compressedStateMap) {
        // how to do this:
        // for each row which leads to a state in the prob1states
        // first get the one-step states:
        // once we have these states it is from these rows
        storm::storage::BitVector oneStepStates = mopmc::graph::getOneStep<ValueType>(
            backwardTransitions, prob1States
        );

        std::cout << "One step states: ";
        for(uint_fast64_t i = 0; i != oneStepStates.size(); ++i){
            if(oneStepStates[i]) {
                std::cout << i << ", ";
            }
        }
        std::cout << "\n";

        std::vector<ValueType> b(dim, 0);

        // For each state if the state is one step from the set of goal states
        // accumulate the values in b
        for (uint_fast64_t i = 0; i < transitionMatrix.rows(); ++i) {
            if (oneStepStates[i]) {
                uint_fast64_t rowStart = transitionMatrix.outerIndexPtr()[i];
                uint_fast64_t rowEnd = transitionMatrix.outerIndexPtr()[i+1];

                for (uint_fast64_t k = rowStart; k < rowEnd; ++k) {
                    uint_fast64_t colIndex = transitionMatrix.innerIndexPtr()[k];
                    if (prob1States[colIndex]) {
                        b[compressedStateMap[i]] += transitionMatrix.valuePtr()[k];
                    }
                }
            }
            
        }
        Eigen::VectorXd eigenVector(b.size());
        for (Eigen::Index i = 0; i < eigenVector.size(); ++i) {
            eigenVector[i] = b[i];
        }
        return eigenVector;
    }

    template<typename ValueType>
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> SparseModelBuilder<ValueType>::solverHelper(
        SpMat const& subMatrix,
        Eigen::Matrix<ValueType, Eigen::Dynamic, 1> const& b){
        
        // construct an Identity matrix
        SpMat identity(subMatrix.rows(), subMatrix.rows());
        for (uint_fast64_t i = 0; i < subMatrix.rows(); ++i) {
            identity.insert(i, i) = 1.0;
        }
        identity.finalize();

        std::cout << "System: \n" << (identity - subMatrix).toDense() << std::endl;
        SpMat Z = identity - subMatrix;

        Eigen::BiCGSTAB<SpMat> solver;
        solver.compute(Z);
        Eigen::Matrix<ValueType, Eigen::Dynamic, 1> x = solver.solve(b);
        std::cout << "Current solution: " << x.transpose() << std::endl; 
        std::cout << "#iterations:     " << solver.iterations() << std::endl;
        std::cout << "estimated error: " << solver.error()      << std::endl;
        /* ... update b ... */
        x = solver.solve(b); // solve again

        std::cout << "Current solution: " << x.transpose() << std::endl; 
        
        return x;
    }

    template class SparseModelBuilder<double>;
}


