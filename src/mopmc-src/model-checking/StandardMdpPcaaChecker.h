//
// Created by thomas on 6/09/23.
//

#ifndef MOPMC_STANDARDMDPPCAACHECKER_H
#define MOPMC_STANDARDMDPPCAACHECKER_H

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
namespace multiobjective {


template<typename SparseModelType>
class StandardMdpPcaaChecker {
    typedef Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> SpMat;
public:
    explicit StandardMdpPcaaChecker(storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<SparseModelType>& preprocessorResult);

    void initialise(storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<SparseModelType>& preprocessorResult);

    void initialiseModelTypeSpecificData(SparseModelType& model);

    void unboundedWeightedPhaseNoEc(storm::Environment const& env,
                                    std::vector<typename SparseModelType::ValueType> const& weightedRewardVector,
                                    std::vector<typename SparseModelType::ValueType> const& weightVector);

    void unboundedWeightedPhase(storm::Environment const& env,
                                std::vector<typename SparseModelType::ValueType> const& weightedRewardVector,
                                std::vector<typename SparseModelType::ValueType> const& weightVector);

    void unboundedIndividualPhase(storm::Environment const& env,
                                  std::vector<std::vector<typename SparseModelType::ValueType>>& rewardModels);

    void computeSchedulerFinitelyOften(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitionMatrix,
                                       storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& backwardTransitions, storm::storage::BitVector const& finitelyOftenChoices,
                                       storm::storage::BitVector safeStates, std::vector<uint64_t>& choices);

    void check(storm::Environment const& env, std::vector<typename SparseModelType::ValueType> const& weightVector);

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

    void transformEcqSolutionToOriginalModel(std::vector<typename SparseModelType::ValueType> const& ecqSolution,
                                             std::vector<uint_fast64_t> const& ecqOptimalChoices,
                                             std::map<uint64_t, uint64_t> const& ecqStateToOptimalMecMap,
                                             std::vector<typename SparseModelType::ValueType>& originalSolution,
                                             std::vector<uint_fast64_t>& originalOptimalChoices) const;

    void multiObjectiveSolver(storm::Environment const& env);

private:

    storm::modelchecker::helper::SparseDeterministicInfiniteHorizonHelper<typename SparseModelType::ValueType>
    createDetInfiniteHorizonHelper(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions) const {
        STORM_LOG_ASSERT(transitions.getRowGroupCount() == this->transitionMatrix.getRowGroupCount(), "Unexpected size of given matrix.");
        return storm::modelchecker::helper::SparseDeterministicInfiniteHorizonHelper<typename SparseModelType::ValueType>(transitions);
    }

    void updateEcQuotient(std::vector<typename SparseModelType::ValueType> const& weightedRewardVector);

    std::vector<storm::modelchecker::multiobjective::Objective<typename SparseModelType::ValueType>> objectives;

    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> makeEigenIdentityMatrix();

    std::vector<uint64_t> computeValidInitialScheduler(
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> &matrix,
        storm::storage::BitVector &rowsWithSumLessOne);

    std::vector<uint64_t> randomScheduler();

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

    void toEigenSparseMatrix();

    void fullEigenSparseMatrix();

    void reduceMatrixToDTMC(Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1> &b,
                            std::vector<uint64_t> const& scheduler);

    storm::storage::SparseMatrix<typename SparseModelType::ValueType> transitionMatrix;
    // The initial state of the considered model
    uint64_t initialState{};

    //Over-approximation of a set of choices that are part of an EC
    storm::storage::BitVector ecChoicesHint;

    bool useEcQuotient = false;

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
    SpMat eigenTransitionMatrix;
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

    std::vector<typename SparseModelType::ValueType> weightedResult;
    // The results for the individual objectives (w.r.t. all states of the model)
    std::vector<std::vector<typename SparseModelType::ValueType>> objectiveResults;
};
}
}
#endif //MOPMC_STANDARDMDPPCAACHECKER_H
