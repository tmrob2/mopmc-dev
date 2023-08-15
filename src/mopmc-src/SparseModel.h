//
// Created by thomas on 6/08/23.
//

#ifndef MOPMC_SPARSEMODEL_H
#define MOPMC_SPARSEMODEL_H
#include <storm/utility/constants.h>
#include <storm/models/sparse/StateLabeling.h>
#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <iostream>

namespace mopmc {
namespace sparse {

enum MatrixType {
    Transition,
    Reward
};

template <typename ValueType>
class SparseModelBuilder {
public:
    typedef Eigen::SparseMatrix<ValueType, Eigen::RowMajor> SpMat;

    SparseModelBuilder() {
        // Intentionally left blank
    }

    void constructMatrixFromTriplet(uint_fast64_t nrow, uint_fast64_t ncol,
                                    std::vector<Eigen::Triplet<ValueType>>& triplets,
                                    MatrixType matrixType) {
        // do nothing
        SpMat mat(nrow, ncol);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        switch (matrixType) {
            case MatrixType::Transition:
                transitionMatrix = mat;
                break;
            case MatrixType::Reward:
                rewardMatrix = mat;
                break;
        }
    };

    SpMat& getSparseMatrix();

    uint_fast64_t& getNumberOfStates();

    void setNumberOfStates(uint_fast64_t numStates);

    uint_fast64_t& getNumberOfChoices();

    void setNumberOfChoices(uint_fast64_t numChoices);

    uint_fast64_t getNumberOfTransitions();

    const storm::storage::BitVector & getInitialStates() const;

    void setNewIndexMap(uint_fast64_t state, uint_fast64_t index);

    std::unordered_map<uint_fast64_t, uint_fast64_t>& getStateActionMapping();

    void setNumberOfTransitions(uint_fast64_t numTransition);

    void setStateLabels(const storm::models::sparse::StateLabeling& sLabels);

    storm::storage::BitVector const& getStates(std::string const& label) const;

    SpMat getBackwardTransitions();

    bool hasLabel(std::string const& label);

    std::pair<SpMat,std::unordered_map<uint_fast64_t, uint_fast64_t>>
        getDTMCSubMatrix(storm::storage::BitVector const& maybeStates);

    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> bVector(
        storm::storage::BitVector const& prob1States,
        SpMat const& backwardTransitions,
        uint_fast64_t dim,
        std::unordered_map<uint_fast64_t, uint_fast64_t>& compressedStateMap);

    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> solverHelper(
        SpMat const& subMatrix,
        Eigen::Matrix<ValueType, Eigen::Dynamic, 1> const& b);
    

private:
    uint_fast64_t numberOfStates;
    uint_fast64_t numberOfTransitions;
    uint_fast64_t numberOfChoices;
    SpMat transitionMatrix;
    SpMat rewardMatrix;
    storm::models::sparse::StateLabeling stateLabels;
    // For MDPs
    std::unordered_map<uint_fast64_t , uint_fast64_t> stateActionMapping;
    std::unordered_map<int, int> enabledActions;
};

}
}

#endif //MOPMC_SPARSEMODEL_H
