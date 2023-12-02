//
// Created by thomas on 6/09/23.
//

#ifndef MOPMC_MOPMCMODELCHECKING_H
#define MOPMC_MOPMCMODELCHECKING_H

#include <storm/storage/SparseMatrix.h>
#include <storm/storage/StateBlock.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <storm/solver/AbstractEquationSolver.h>
#include <storm/modelchecker/helper/infinitehorizon/SparseDeterministicInfiniteHorizonHelper.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include "SparseMultiObjective.h"

namespace mopmc {
namespace multiobjective {


template<typename SparseModelType>
class MOPMCModelChecking : protected mopmc::multiobjective::MultiObjectiveModel<SparseModelType> {
public:

    //! We follow the storm class and method naming convention. Technically this class does not exist in the Storm code
//! base, however, the derived class for MDPs is something similar to this
    explicit MOPMCModelChecking(storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<SparseModelType> &preprocessorResult)
    {
        this->objectives = preprocessorResult.objectives;
        this->initialise(preprocessorResult);
    }
    typedef Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> SpMat;

    void unboundedWeightedPhase(storm::Environment const& env,
                                std::vector<typename SparseModelType::ValueType> const& weightedRewardVector,
                                std::vector<typename SparseModelType::ValueType> const& weightVector);

    //void unboundedIndividualPhase(storm::Environment const& env,
    //                              std::vector<std::vector<typename SparseModelType::ValueType>>& rewardModels);

    //void unboundedIndividualPhase(storm::Environment const& env,
    //                              std::vector<typename SparseModelType::ValueType> const& weightVector);

    void unboundedIndividualPhase(storm::Environment const& env,
                                  std::vector<std::vector<typename SparseModelType::ValueType>>& rewardModels,
                                  std::vector<typename SparseModelType::ValueType> const& weightVector);

    void check(storm::Environment const& env, std::vector<typename SparseModelType::ValueType> const& weightVector);

    void multiObjectiveSolver(storm::Environment const& env);

    //GS: add public getters. :SG
public:
    const std::vector<std::vector<typename SparseModelType::ValueType>> &getObjectiveResults() const {
        return this->objectiveResults;
    }
    uint64_t getInitialState() const {
        return this->initialState;
    }

    std::vector<typename SparseModelType::ValueType> getAuxStateValues() {
        return this->ecQuotient->auxStateValues;
    }

    std::vector<typename SparseModelType::ValueType> getAuxChoiceValues() {
        return this->ecQuotient->auxChoiceValues;
    }

    std::vector<uint64_t> getAuxRowGroupIndices(){
        return this->ecQuotient->matrix.getRowGroupIndices();
    }


    std::vector<uint_fast64_t> getEcqToOriginalChoiceMapping() {
        return this->ecQuotient->ecqToOriginalChoiceMapping;
    }

   boost::optional<typename mopmc::multiobjective::MultiObjectiveModel<SparseModelType>::EcQuotient> getEcQuotient() {
        return this->ecQuotient;
    }

    void toEigenSparseMatrixPub() {
        this->toEigenSparseMatrix();
    }

    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> getEigenTransitionMatrix(){
        //this->toEigenSparseMatrix();
        return this->eigenTransitionMatrix;
    }

    void updateEcQuotientPub(std::vector<typename SparseModelType::ValueType> const &weightedRewardVector) {
        this->updateEcQuotient(weightedRewardVector);
    }

    storm::storage::SparseMatrix<typename SparseModelType::ValueType> getTransitionMatrix() {
        return this->transitionMatrix;
    }

    std::vector<std::vector<typename SparseModelType::ValueType>> getActionRewards() {
        return this->actionRewards;
    }

    std::vector<std::vector<typename SparseModelType::ValueType>> getStateRewards() {
        return this->stateRewards;
    }

    std::vector<uint64_t > computeValidInitialSchedulerPubNoArg() {
        return this->computeValidInitialScheduler(this->ecQuotient->matrix, this->ecQuotient->rowsWithSumLessOne);
    }


private:

    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> makeEigenIdentityMatrix();

    //std::vector<uint64_t> randomScheduler(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& matrix);

    void toEigenSparseMatrix();

    void reduceMatrixToDTMC(Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1> &b,
                            std::vector<uint64_t> const& scheduler);

    // Members
    SpMat eigenTransitionMatrix;
    SpMat dtmcTransitionMatrix;
    std::vector<typename SparseModelType::ValueType> weightedResult;
    // The results for the individual objectives (w.r.t. all states of the model)

};
}
}
#endif //MOPMC_MOPMCMODELCHECKING_H
