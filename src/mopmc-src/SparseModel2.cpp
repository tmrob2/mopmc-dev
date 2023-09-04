//
// Created by thomas on 2/09/23.
//
#include <storm/models/sparse/Mdp.h>
#include "SparseModel2.h"
#include "mopmc-src/model-checking/MultiObjectivePreprocessor.h"

namespace mopmc{
namespace sparsemodel{

template <typename SparseModelType>
SparseModelBuilder<SparseModelType>::SparseModelBuilder(typename mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType& result)
:
numberOfStates(result.preprocessedModel->getNumberOfStates()),
numberOfTransitions(result.preprocessedModel->getNumberOfTransitions()),
numberOfChoices(result.preprocessedModel->getNumberOfChoices()),
rowGroupIndices(result.preprocessedModel->getTransitionMatrix().getRowGroupIndices()),
initialStates(result.preprocessedModel->getInitialStates()) {
      std::cout << "transition matrix rows: " << result.preprocessedModel->getTransitionMatrix().getRowCount() << "\n";
      toEigenSparseMatrix(result.preprocessedModel->getTransitionMatrix());
      convertRewardModelsToSparseMatrix(this->transitionMatrix.rows(),
                                      result.preprocessedModel->getNumberOfRewardModels(),
                                      result.preprocessedModel->getRewardModels());
      setReverseStateActionMap(result.preprocessedModel->getNumberOfStates());
      makeIdentityMatrix(result.preprocessedModel->getNumberOfStates());
}

template<typename SparseModelType>
void SparseModelBuilder<SparseModelType>::toEigenSparseMatrix(
    storm::storage::SparseMatrix<typename SparseModelType::ValueType> matrix){
    std::vector<Eigen::Triplet<typename SparseModelType::ValueType>> triplets;
    triplets.reserve(matrix.getNonzeroEntryCount());

    for(uint_fast64_t row = 0; row < matrix.getRowCount(); ++row) {
        for(auto element : matrix.getRow(row)) {
            triplets.emplace_back(row, element.getColumn(), element.getValue());
            element.setValue(0.0);
        }
    }

    matrix.dropZeroEntries();

    Eigen::SparseMatrix<typename SparseModelType::ValueType,  Eigen::RowMajor> result =
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>(
            matrix.getRowCount(), matrix.getColumnCount()
        );
    result.setFromTriplets(triplets.begin(), triplets.end());
    this->transitionMatrix = result;
}

template<typename SparseModelType>
void SparseModelBuilder<SparseModelType>::convertRewardModelsToSparseMatrix(
    uint_fast64_t numberOfRows,
    uint_fast64_t numberOfRewardModels,
    std::unordered_map<std::string,storm::models::sparse::StandardRewardModel<typename SparseModelType::ValueType>>& stormRewardModels) {
    std::vector<std::string> names(numberOfRewardModels);
    std::vector<Eigen::Triplet<typename SparseModelType::ValueType>> triplets;
    uint_fast64_t jj = 0;
    for(auto& pair : stormRewardModels) {
        names[jj] = pair.first;
        if(pair.second.hasStateActionRewards()) {
            uint_fast64_t ii = 0;
            for(typename SparseModelType::ValueType val : pair.second.getStateActionRewardVector()) {
                if (!storm::utility::isZero(val)) {
                    // emplace it into the sparseRewardsMatrix
                    triplets.emplace_back(ii, jj, val);
                }
                ++ii;
            }
        } else if (pair.second.hasStateRewards()) {
            throw std::runtime_error("Not handled");
        }
        ++jj;
    }
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> result =
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>(
            numberOfRows, numberOfRewardModels);
    result.setFromTriplets(triplets.begin(), triplets.end());
    this->rewardModels = result;
    this->rewardModelNames = names;
}

template<typename SparseModelType>
void SparseModelBuilder<SparseModelType>::setReverseStateActionMap(uint_fast64_t nStates) {
    for(uint_fast64_t state = 0; state < nStates; ++state){
        uint_fast64_t first = this->rowGroupIndices[state];
        uint_fast64_t last = this->rowGroupIndices[state + 1];
        for (uint_fast64_t action = first; action <= last; ++action) {
            this->reverseStateActionMapping[action] = state;
        }
    }
}

template<typename SparseModelType>
typename SparseModelBuilder<SparseModelType>::SpMat& SparseModelBuilder<SparseModelType>::getTransitionMatrix() {
    return this->transitionMatrix;
}

template<typename SparseModelType>
typename SparseModelBuilder<SparseModelType>::SpMat& SparseModelBuilder<SparseModelType>::getRewardMatrix() {
    return this->rewardModels;
}

template<typename SparseModelType>
std::vector<std::string>& SparseModelBuilder<SparseModelType>::getRewardModelNames() {
    return this->rewardModelNames;
}

template<typename SparseModelType>
uint_fast64_t SparseModelBuilder<SparseModelType>::getNumberOfStates() {
    return this->numberOfStates;
}

template<typename SparseModelType>
uint_fast64_t SparseModelBuilder<SparseModelType>::getNumberOfTransitions() {
    return this->numberOfTransitions;
}

template<typename SparseModelType>
std::vector<uint_fast64_t>& SparseModelBuilder<SparseModelType>::getRowGroupIndices() {
    return this->rowGroupIndices;
}

template<typename SparseModelType>
std::unordered_map<uint_fast64_t, uint_fast64_t>& SparseModelBuilder<SparseModelType>::getReverseStateActionMapping() {
    return this->reverseStateActionMapping;
}

template<typename SparseModelType>
storm::storage::BitVector& SparseModelBuilder<SparseModelType>::getInitialStates() {
    return this->initialStates;
}

template<typename SparseModelType>
void SparseModelBuilder<SparseModelType>::makeIdentityMatrix(uint_fast64_t nStates) {
    typename mopmc::sparsemodel::SparseModelBuilder<SparseModelType>::SpMat idMatrix(nStates, nStates);
    for(uint_fast64_t state; state < nStates; ++state) {
        idMatrix.insert(state, state) = 1.0;
    }
    idMatrix.finalize();
    this->identityMatrix = idMatrix;
}

template <typename SparseModelType>
typename mopmc::sparsemodel::SparseModelBuilder<SparseModelType>::SpMat mopmc::sparsemodel::SparseModelBuilder<SparseModelType>::createSubMatrix(
    std::vector<uint_fast64_t>& policy) {
    // loop over each of the states and select the rows from the transition matrix and the
    // rewards matrix which correspond to actions(indices) in the policy
    SpMat subTransitionMatrix(this->numberOfStates, this->numberOfStates);
    for(uint_fast64_t state = 0; state < this->numberOfStates; ++state) {
        uint_fast64_t selectedAction = policy[state];
        typename SpMat::InnerIterator it(this->transitionMatrix, selectedAction);
        for(; it ; ++it) {
            subTransitionMatrix.insert(state, it.col()) = it.value();
        }
    }
    subTransitionMatrix.makeCompressed();
}

template class SparseModelBuilder<storm::models::sparse::Mdp<double>>;
}
}
