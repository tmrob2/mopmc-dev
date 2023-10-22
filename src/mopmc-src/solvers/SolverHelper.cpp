//
// Created by thomas on 26/09/23.
//

#include "SolverHelper.h"
#include <storm/models/sparse/Mdp.h>

namespace mopmc::solver::helper {
    template<typename SparseModelType>
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> eigenInducedTransitionMatrix(
            storm::storage::SparseMatrix<typename SparseModelType::ValueType> &fullTransitionSystem,
            std::vector<uint64_t> &scheduler,
            std::vector<uint_fast64_t> const &rowGroupIndices
    ) {

        assert(scheduler.size() == fullTransitionSystem.getColumnCount());
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> subMatrix(scheduler.size(),
                                                                                            scheduler.size());
        for (uint_fast64_t state = 0; state < fullTransitionSystem.getRowGroupCount(); ++state) {
            auto const &actionStart = fullTransitionSystem.getRowGroupIndices()[state];
            for (auto element : fullTransitionSystem.getRow(actionStart + scheduler[state])) {
                //std::cout << " rows, cols: " << state << ", " << element.getColumn() << "\n";
                subMatrix.insert(state, element.getColumn()) = element.getValue();
            }
        }
        subMatrix.makeCompressed();
        //std::cout << subMatrix.toDense() << std::endl;
        return subMatrix;
    }

    template <typename SparseModelType>
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> eigenInducedTransitionMatrix(
            Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> &fullTransitionSystem,
            std::vector<uint64_t>& scheduler,
            std::vector<uint_fast64_t> const& rowGroupIndices) {

    }

    template<typename ValueType>
    void inducedRewards(std::vector<ValueType> &b, std::vector<ValueType> &subB,
                        std::vector<uint64_t> &scheduler, std::vector<uint_fast64_t> const &rowGroupIndices) {
        for (uint_fast64_t state = 0; state < scheduler.size(); ++state) {
            auto const &actionStart = rowGroupIndices[state];
            //std::cout << "state " << state << " action " << actionStart;
            subB[state] = b[actionStart + scheduler[state]];
        }
    }

    template<typename SparseModelType>
    storm::storage::BitVector performProbGreater0(
            Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> const& backwardTransitions,
            storm::storage::BitVector const& phiStates,
            storm::storage::BitVector const& psiStates) {
        // Prepare the resulting bit vector
        uint_fast64_t numberOfStates = phiStates.size();
        storm::storage::BitVector statesWithProbabilityGreater0(numberOfStates);

        // Add all psi states as they already satisfy the condition
        statesWithProbabilityGreater0 |= psiStates;

        // Initialise the state used for the DFS with the states
        std::vector<uint_fast64_t> stack(psiStates.begin(), psiStates.end());

        // Initialise the stack for the step bound if the steps is bnouinded
        std::vector<uint_fast64_t> stepStack;
        std::vector<uint_fast64_t> remainingSteps;

        // Perform the actual DFS
        uint_fast64_t currentState, currentStepBound;
        while (!stack.empty()) {
            currentState = stack.back();
            stack.pop_back();

            // iterate over the state
            for (typename Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>::InnerIterator it(backwardTransitions, currentState); it; ++it) {
                if (phiStates[it.col()] && !statesWithProbabilityGreater0.get(it.col())) {
                    statesWithProbabilityGreater0.set(it.col(), true);
                    stack.push_back(it.col());
                }
            }
        }

        return statesWithProbabilityGreater0;;
    }

    template<typename SparseModelType>
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> constructSubMatrix(
            Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>& dtmc,
            storm::storage::BitVector& maybeStates,
            std::unordered_map<uint_fast64_t, uint_fast64_t>& subMatMap
    ){
        uint_fast64_t  subMatrixSize = maybeStates.getNumberOfSetBits();

        // create a hashmap which stors the state to its compressed value
        uint_fast64_t newIndex = 0;
        for(int i = 0; i != maybeStates.size() ; ++i) {
            if (maybeStates[i]) {
                subMatMap.emplace(i, newIndex);
                ++newIndex;
            }
        }

        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> subMatrix(subMatrixSize, subMatrixSize);

        for (int k = 0; k < dtmc.outerSize(); ++k) {
            for(typename Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>::InnerIterator it(dtmc, k); it; ++it) {
                if(maybeStates[it.row()] && maybeStates[it.col()]) {
                    subMatrix.insert(subMatMap[it.row()], subMatMap[it.col()]) = it.value();
                }
            }
        }
        subMatrix.makeCompressed();
        return subMatrix;
    }

    template <typename SparseModelType>
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> makeIdentity(uint_fast64_t n){
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> I(n, n);
        for (uint_fast64_t i = 0; i < n; ++i) {
            I.insert(i, i) = static_cast<typename SparseModelType::ValueType>(1.0);
        }
        return I;
    }

    template<typename SparseModelType>
    void makeRhs(std::vector<typename SparseModelType::ValueType>& b,
                 std::vector<typename SparseModelType::ValueType>const& deterministicStateRewards,
                 //std::vector<uint_fast64_t>const& rowGroupIndices,
                 storm::storage::BitVector& maybeStates
                 //std::vector<uint_fast64_t>& scheduler
                 ){
        uint_fast64_t cntr = 0;
        for (uint_fast64_t i = 0; i < deterministicStateRewards.size(); ++i) {
            if (maybeStates[i]) {
                b[cntr] = deterministicStateRewards[i];
                ++cntr;
            }
        }
    }

    // Explicit instantiation --------------------
    template Eigen::SparseMatrix<double, Eigen::RowMajor>
    eigenInducedTransitionMatrix<storm::models::sparse::Mdp<double>>(
            storm::storage::SparseMatrix<double> &fullTransitionSystem,
            std::vector<uint64_t> &scheduler,
            std::vector<uint_fast64_t> const &rowGroupIndices);

    template void inducedRewards(std::vector<double> &b, std::vector<double> &subB,
                                 std::vector<uint64_t> &scheduler,
                                 std::vector<uint_fast64_t> const &rowGroupIndices);

    template storm::storage::BitVector performProbGreater0<storm::models::sparse::Mdp<double>>(
            Eigen::SparseMatrix<double, Eigen::RowMajor> const& backwardTransitions,
            storm::storage::BitVector const& phiStates,
            storm::storage::BitVector const& psiStates);

    template Eigen::SparseMatrix<double, Eigen::RowMajor> constructSubMatrix<storm::models::sparse::Mdp<double>>(
            Eigen::SparseMatrix<double, Eigen::RowMajor> & dtmc,
            storm::storage::BitVector& maybeStates,
            std::unordered_map<uint_fast64_t, uint_fast64_t>& subMatMap
    );

    template Eigen::SparseMatrix<double, Eigen::RowMajor> makeIdentity<storm::models::sparse::Mdp<double>>(uint_fast64_t n);

    template void makeRhs<storm::models::sparse::Mdp<double>>(
            std::vector<double>& b,
            std::vector<double>const& deterministicStateRewards,
            //std::vector<uint_fast64_t>const& rowGroupIndices,
            storm::storage::BitVector& maybeStates);
            //std::vector<uint_fast64_t>& scheduler);
}