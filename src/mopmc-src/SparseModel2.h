//
// Created by thomas on 2/09/23.
//

#ifndef MOPMC_SPARSEMATRIX2_H
#define MOPMC_SPARSEMATRIX2_H

#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <storm/models/sparse/StateLabeling.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <boost/optional.hpp>
#include <storm/storage/BitVector.h>
#include "mopmc-src/model-checking/MultiObjectivePreprocessor.h"

namespace mopmc {
namespace sparsemodel {

template <typename SparseModelType>
class SparseModelBuilder {
public:

    typedef Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> SpMat;
    /* Conversion to the data structure needed for the convex query:
     *  TODO: An eigen transition matrix can be constructed using a code template from storm
     *   EigenAdapter which consumes the sparseMatrix and creates an Eigen Sparse Matrix
     */
    explicit SparseModelBuilder(typename mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType& model);

    SpMat& getTransitionMatrix();

    SpMat& getRewardMatrix();

    std::vector<std::string>& getRewardModelNames();

    uint_fast64_t getNumberOfStates();

    uint_fast64_t getNumberOfTransitions();

    std::vector<uint_fast64_t>& getRowGroupIndices();

    std::unordered_map<uint_fast64_t, uint_fast64_t>& getReverseStateActionMapping();

    storm::storage::BitVector& getInitialStates();

    SpMat createSubMatrix(std::vector<uint_fast64_t>& policy);

private:
    void makeIdentityMatrix(uint_fast64_t nStates);
    /*!
     * Converts a storm transition matrix (Sparse Matrix) into an Eigen Sparse matrix and
     * deletes the storm sparse matrix structure afterwards
     * @param matrix
     */
    void toEigenSparseMatrix(storm::storage::SparseMatrix<typename SparseModelType::ValueType> matrix);

    /*!
     * Converts a set of storm reward models into a sparse matrix
    */
    void convertRewardModelsToSparseMatrix(
        uint_fast64_t numberOfRows,
        uint_fast64_t numberOfRewardModels,
        std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<typename SparseModelType::ValueType>>& stormRewardModels);

    void setReverseStateActionMap(uint_fast64_t getNumberOfStates);

    // TODO: do we need to check which states get remapped to some larger system?
    storm::storage::BitVector initialStates;
    uint_fast64_t numberOfStates;
    uint_fast64_t numberOfTransitions;
    uint_fast64_t numberOfChoices;
    SpMat transitionMatrix;
    std::vector<std::string> rewardModelNames;
    SpMat rewardModels;
    storm::models::sparse::StateLabeling stateLabels;
    std::vector<uint_fast64_t> rowGroupIndices;
    std::unordered_map<uint_fast64_t, uint_fast64_t> reverseStateActionMapping;
    SpMat identityMatrix;
};

}
}

#endif //MOPMC_SPARSEMATRIX2_H
