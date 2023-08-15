//
// Created by thomas on 8/08/23.
//

#include "SparseModel.h"
#include "model-checking/ModelCheckingUtils.h"
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
    const storm::storage::BitVector& SparseModelBuilder<ValueType>::getInitialStates() const {
        return this -> getStates("init");
    }

    template <typename ValueType>
    void SparseModelBuilder<ValueType>::setNewIndexMap(uint_fast64_t state, uint_fast64_t index) {
        this -> stateActionMapping[index] = state;
    }

    template <typename ValueType>
    std::unordered_map<uint_fast64_t, uint_fast64_t>& SparseModelBuilder<ValueType>::getStateActionMapping() {
        return this -> stateActionMapping;
    }

    template <typename ValueType>
    std::pair<typename SparseModelBuilder<ValueType>::SpMat, std::unordered_map<uint_fast64_t, uint_fast64_t>>
            SparseModelBuilder<ValueType>::getDTMCSubMatrix(
        const storm::storage::BitVector &maybeStates) {

        // keep track of the initial state mappings from full to sub-matrix.

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
        storm::storage::BitVector oneStepStates = mopmc::sparseutils::getOneStep<ValueType>(
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


