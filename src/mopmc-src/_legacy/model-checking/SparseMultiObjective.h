//
// Created by thomas on 20/08/23.
//

#ifndef MOPMC_SPARSEMULTIOBJECTIVE_H
#define MOPMC_SPARSEMULTIOBJECTIVE_H
#include <iostream>
#include <memory>
#include <storm/environment/Environment.h>
#include <storm/logic/MultiObjectiveFormula.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/StateBlock.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <storm/solver/AbstractEquationSolver.h>
#include <storm/modelchecker/helper/infinitehorizon/SparseDeterministicInfiniteHorizonHelper.h>

namespace mopmc {
namespace multiobjective{

template<typename SparseModelType>
class MultiObjectiveModel {
public:
    MultiObjectiveModel() = default;

    void initialise(storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<SparseModelType>& preprocessorResult);

    void initialiseModelTypeSpecificData(SparseModelType& model);


    void computeSchedulerFinitelyOften(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitionMatrix,
                                       storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& backwardTransitions, storm::storage::BitVector const& finitelyOftenChoices,
                                       storm::storage::BitVector safeStates, std::vector<uint64_t>& choices);

    boost::optional<typename SparseModelType::ValueType> computeWeightedResultBound(
            bool lower, std::vector<typename SparseModelType::ValueType> const& weightVector, storm::storage::BitVector const& objectiveFilter) const;

    void setBoundsToSolver(storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver,
                           bool requiresLower,
                           bool requiresUpper,
                           uint64_t objIndex,
                           storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions,
                           storm::storage::BitVector const& rowsWithSumLessOne,
                           std::vector<typename SparseModelType::ValueType> const& rewards) const;

    void setBoundsToSolver(storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver,
                           bool requiresLower,
                           bool requiresUpper,
                           std::vector<typename SparseModelType::ValueType> const& weightVector,
                           storm::storage::BitVector const& objectiveFilter,
                           storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions,
                           storm::storage::BitVector const& rowsWithSumLessOne,
                           std::vector<typename SparseModelType::ValueType> const& rewards);

    void computeAndSetBoundsToSolver(storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver,
                                     bool requiresLower,
                                     bool requiresUpper,
                                     storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions,
                                     storm::storage::BitVector const& rowsWithSumLessOne,
                                     std::vector<typename SparseModelType::ValueType> const& rewards) const;

private:

    storm::modelchecker::helper::SparseDeterministicInfiniteHorizonHelper<typename SparseModelType::ValueType>
    createDetInfiniteHorizonHelper(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions) const {
        STORM_LOG_ASSERT(transitions.getRowGroupCount() == this->transitionMatrix.getRowGroupCount(), "Unexpected size of given matrix.");
        return storm::modelchecker::helper::SparseDeterministicInfiniteHorizonHelper<typename SparseModelType::ValueType>(transitions);
    }

    void computeSchedulerProb1(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitionMatrix,
                               storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& backwardTransitions,
                               storm::storage::BitVector const& consideredStates,
                               storm::storage::BitVector const& statesToReach,
                               std::vector<uint64_t>& choices,
                               storm::storage::BitVector const* allowedChoices = nullptr) const;

    void computeSchedulerProb0(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitionMatrix,
                               storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& backwardTransitions,
                               storm::storage::BitVector const& consideredStates, storm::storage::BitVector const& statesToAvoid,
                               storm::storage::BitVector const& allowedChoices, std::vector<uint64_t>& choices) const;

protected:

    void updateEcQuotient(std::vector<typename SparseModelType::ValueType> const& weightedRewardVector);

    void transformEcqSolutionToOriginalModel(std::vector<typename SparseModelType::ValueType> const& ecqSolution,
                                             std::vector<uint_fast64_t> const& ecqOptimalChoices,
                                             std::map<uint64_t, uint64_t> const& ecqStateToOptimalMecMap,
                                             std::vector<typename SparseModelType::ValueType>& originalSolution,
                                             std::vector<uint_fast64_t>& originalOptimalChoices) const;

    std::vector<uint64_t> computeValidInitialScheduler(
            storm::storage::SparseMatrix<typename SparseModelType::ValueType> &matrix,
            storm::storage::BitVector &rowsWithSumLessOne);

    // The initial state of the considered model
    uint64_t initialState{};

    //Over-approximation of a set of choices that are part of an EC
    storm::storage::BitVector ecChoicesHint;

    bool useEcQuotient = false;

    // Objectives
    std::vector<std::vector<typename SparseModelType::ValueType>> objectiveResults;
    std::vector<storm::modelchecker::multiobjective::Objective<typename SparseModelType::ValueType>> objectives;

    // The actions that have reward assigned for at least one objective without
    // upper time-bound
    storm::storage::BitVector actionsWithoutRewardInUnboundedPhase;
    // The states for which there is a scheduler yielding reward 0 for each total
    // reward objective
    storm::storage::BitVector totalReward0EStates;
    // stores the state action reward for each objective.
    std::vector<std::vector<typename SparseModelType::ValueType>> actionRewards;

    // These are only relevant for LRA objectives for MAs (otherwise, they appear within the action rewards). For other objectives/models, the corresponding
    // vector will be empty.
    std::vector<std::vector<typename SparseModelType::ValueType>> stateRewards;


    // stores the indices of the objectives for which we need to compute the
    // long run average values
    storm::storage::BitVector lraObjectives;

    // stores the indices of the objectives for which there is no upper time bound
    storm::storage::BitVector objectivesWithNoUpperTimeBound;

    struct LraMecDecomposition {
        storm::storage::MaximalEndComponentDecomposition<typename SparseModelType::ValueType> mecs;
        std::vector<typename SparseModelType::ValueType> auxMecValues;
    };
    boost::optional<LraMecDecomposition> lraMecDecomposition;
    typename SparseModelType::ValueType epsilon;

    std::vector<uint_fast64_t> pi;

    // The scheduler choices that optimize the weighted rewards of unbounded objectives.
    std::vector<uint64_t> optimalChoices;

    // Memory for the solution of the most recent call of check(..)
    // becomes true after the first call of check(..)
    bool checkHasBeenCalled {};
    // The distances are stored as a (possibly negative) offset that has to be added (+) to to the objectiveResults.
    std::vector<typename SparseModelType::ValueType> offsetsToUnderApproximation;
    std::vector<typename SparseModelType::ValueType> offsetsToOverApproximation;

    struct EcQuotient {
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> matrix;
        std::vector<uint_fast64_t> ecqToOriginalChoiceMapping;
        std::vector<uint_fast64_t> originalToEcqStateMapping;
        std::vector<storm::storage::FlatSetStateContainer> ecqToOriginalStateMapping;
        storm::storage::BitVector ecqStayInEcChoices;
        storm::storage::BitVector origReward0Choices;       // includes total and LRA rewards
        storm::storage::BitVector origTotalReward0Choices;  // considers just total rewards
        storm::storage::BitVector rowsWithSumLessOne;

        std::vector<typename SparseModelType::ValueType> auxStateValues;
        std::vector<typename SparseModelType::ValueType> auxChoiceValues;
    };

    std::vector<typename SparseModelType::ValueType> auxStateValues;
    std::vector<typename SparseModelType::ValueType> auxChoiceValues;

    boost::optional<EcQuotient> ecQuotient;

    storm::storage::SparseMatrix<typename SparseModelType::ValueType> transitionMatrix;
};

template<typename SparseModelType>
//std::unique_ptr<storm::modelchecker::CheckResult>
void performMultiObjectiveModelChecking(
    storm::Environment env,
    SparseModelType& model,
    storm::logic::MultiObjectiveFormula const& formula);
    }
}




#endif //MOPMC_SPARSEMULTIOBJECTIVE_H
