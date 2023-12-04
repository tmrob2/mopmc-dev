//
// Created by thomas on 20/08/23.
//
#include <iostream>
#include <memory>
#include <storm/modelchecker/multiobjective/multiObjectiveModelChecking.h>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/modelchecker/multiobjective/pcaa/SparsePcaaQuery.h>
#include "MOPMCModelChecking.h"
#include "MultiObjectivePreprocessor.h"
#include <storm/models/sparse/Mdp.h>

#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include <storm/transformer/EndComponentEliminator.h>
#include <storm/transformer/GoalStateMerger.h>
#include <storm/utility/vector.h>
#include <set>
#include "../solvers/InducedEquationSolver.h"
#include <storm/modelchecker/prctl/helper/DsMpiUpperRewardBoundsComputer.h>
#include <storm/modelchecker/prctl/helper/BaierUpperRewardBoundsComputer.h>
#include <storm/solver/LinearEquationSolver.h>
#include "../solvers/ConvexQuery.h"
#include "../solvers/WarmUp.h"

namespace mopmc {
namespace multiobjective {

template<typename SparseModelType>
//std::unique_ptr<storm::modelchecker::CheckResult>
void performMultiObjectiveModelChecking(
        storm::Environment env,
        SparseModelType &model,
        storm::logic::MultiObjectiveFormula const &formula) {

    typename mopmc::multiobjective::SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType result =
            mopmc::multiobjective::SparseMultiObjectivePreprocessor<SparseModelType>::preprocess(
                env,
                model,
                formula);

    // ok now we start experimenting with the result
    std::ostream& outputStream = std::cout;

    result.preprocessedModel->printModelInformationToStream(outputStream);

    std::cout << "Trivial objectives: " << (result.containsOnlyTrivialObjectives() ? "yes" : "no") << "\n";

    mopmc::multiobjective::MOPMCModelChecking<SparseModelType> mdpChecker(result);

    // warmup the GPU if necessary
    mopmc::kernels::launchWarmupKernel();

    std::vector<typename SparseModelType::ValueType> w = {0.5, 0.5};

    //mdpChecker.multiObjectiveSolver(env);
    mdpChecker.check(env, w);
}

//! This is a storm function for initialising the multi-objective model and setting up the required data structures.
template<typename SparseModelType>
void MultiObjectiveModel<SparseModelType>::initialise(
        storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<SparseModelType> &preprocessorResult) {
    auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<SparseModelType>::analyze(preprocessorResult);

    if (rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::Infinite){
        throw std::runtime_error("There is no Pareto optimal scheduler that yields finite reward for all objectives. This is not supported.");
    }
    if (!rewardAnalysis.totalRewardLessInfinityEStates){
        throw std::runtime_error("The set of states with reward < infinity for some scheduler has not been computed during preprocessing.");
    }
    if(!preprocessorResult.containsOnlyTrivialObjectives()) {
        throw std::runtime_error("At least one objective was not reduced to an expected (long run, total or cumulative) reward objective during preprocessing. "
                                 "This is not supported by the considered weight vector checker.");
    }
    if(preprocessorResult.preprocessedModel->getInitialStates().getNumberOfSetBits() > 1){
        throw std::runtime_error("The model has multiple initial states.");
    }
    storm::storage::BitVector maybeStates = rewardAnalysis.totalRewardLessInfinityEStates.get() & ~rewardAnalysis.reward0AStates;
    storm::storage::BitVector finiteTotalRewardChoices = preprocessorResult.preprocessedModel->getTransitionMatrix().getRowFilter(
            rewardAnalysis.totalRewardLessInfinityEStates.get(), rewardAnalysis.totalRewardLessInfinityEStates.get());
    std::set<std::string> relevantRewardModels;
    for (auto const& obj : this->objectives) {
        obj.formula->gatherReferencedRewardModels(relevantRewardModels);
    }
    storm::transformer::GoalStateMerger<SparseModelType> merger(*preprocessorResult.preprocessedModel);
    auto mergerResult =
            merger.mergeTargetAndSinkStates(maybeStates, rewardAnalysis.reward0AStates, storm::storage::BitVector(maybeStates.size(), false),
                                            std::vector<std::string>(relevantRewardModels.begin(), relevantRewardModels.end()), finiteTotalRewardChoices);

    // Initialize data specific for the considered model type
    initialiseModelTypeSpecificData(*mergerResult.model);

    // Initialise general data of the model
    transitionMatrix = std::move(mergerResult.model->getTransitionMatrix());
    initialState = *mergerResult.model->getInitialStates().begin();
    totalReward0EStates = rewardAnalysis.totalReward0EStates % maybeStates;
    if (mergerResult.targetState) {
        // There is an additional state in the result
        totalReward0EStates.resize(totalReward0EStates.size() + 1, true);

        // The overapproximation for the possible ec choices consists of the states that can reach the target states with prob. 0 and the target state itself.
        storm::storage::BitVector targetStateAsVector(transitionMatrix.getRowGroupCount(), false);
        targetStateAsVector.set(*mergerResult.targetState, true);
        ecChoicesHint = transitionMatrix.getRowFilter(
                storm::utility::graph::performProb0E(transitionMatrix, transitionMatrix.getRowGroupIndices(), transitionMatrix.transpose(true),
                                                     storm::storage::BitVector(targetStateAsVector.size(), true), targetStateAsVector));
        ecChoicesHint.set(transitionMatrix.getRowGroupIndices()[*mergerResult.targetState], true);
    } else {
        ecChoicesHint = storm::storage::BitVector(transitionMatrix.getRowCount(), true);
    }

    // set data for unbounded objectives
    lraObjectives = storm::storage::BitVector(this->objectives.size(), false);
    objectivesWithNoUpperTimeBound = storm::storage::BitVector(this->objectives.size(), false);
    actionsWithoutRewardInUnboundedPhase = storm::storage::BitVector(transitionMatrix.getRowCount(), true);
    for (uint_fast64_t objIndex = 0; objIndex < this->objectives.size(); ++objIndex) {
        auto const& formula = *this->objectives[objIndex].formula;
        if (formula.getSubformula().isTotalRewardFormula()) {
            objectivesWithNoUpperTimeBound.set(objIndex, true);
            actionsWithoutRewardInUnboundedPhase &= storm::utility::vector::filterZero(actionRewards[objIndex]);
        }
        if (formula.getSubformula().isLongRunAverageRewardFormula()) {
            lraObjectives.set(objIndex, true);
            objectivesWithNoUpperTimeBound.set(objIndex, true);
        }
    }

    // Set data for LRA objectives (if available)
    if (!lraObjectives.empty()) {
        throw std::runtime_error("This framework does not handle LRA");
    }

    // initialize data for the results
    checkHasBeenCalled = false;
    objectiveResults.resize(this->objectives.size());
    offsetsToUnderApproximation.resize(this->objectives.size(), storm::utility::zero<typename SparseModelType::ValueType>());
    offsetsToOverApproximation.resize(this->objectives.size(), storm::utility::zero<typename SparseModelType::ValueType>());
    optimalChoices.resize(transitionMatrix.getRowGroupCount(), 0);

    // Print some statistics (if requested)
    std::cout << "Weight Vector Checker Statistics:\n";
    std::cout << "Final preprocessed model has " << transitionMatrix.getRowGroupCount() << " states.\n";
    std::cout << "Final preprocessed model has " << transitionMatrix.getRowCount() << " actions.\n";
    if (lraMecDecomposition) {
        std::cout << "Found " << lraMecDecomposition->mecs.size() << " end components that are relevant for LRA-analysis.\n";
        uint64_t numLraMecStates = 0;
        for (auto const& mec : this->lraMecDecomposition->mecs) {
            numLraMecStates += mec.size();
        }
        std::cout << numLraMecStates << " states lie on such an end component.\n";
    }
    std::cout << std::endl;

}


//! This is a storm function - its purpose is just to set up some problem specific information about the model
//! and is essentially an admin function.
template <typename SparseModelType>
void MultiObjectiveModel<SparseModelType>::initialiseModelTypeSpecificData(SparseModelType &model) {
    // set the state action rewards. Also do some sanity checks on the objectives
    this->actionRewards.resize(this->objectives.size());
    std::cout << "MDP model setup -> Objectives: " <<this->objectives.size() << "\n";
    for(uint_fast64_t objIndex = 0; objIndex < this->objectives.size(); ++objIndex){
        auto const& formula = *this->objectives[objIndex].formula;
        if (!(formula.isRewardOperatorFormula() && formula.asRewardOperatorFormula().hasRewardModelName())){
            std::stringstream ss;
            ss << "Unexpected type of operator formula: " << formula;
            throw std::runtime_error(ss.str());
        }
        if (formula.getSubformula().isCumulativeRewardFormula()) {
            auto const& cumulativeRewardFormula = formula.getSubformula().asCumulativeRewardFormula();
            if (!(!cumulativeRewardFormula.isMultiDimensional() && !cumulativeRewardFormula.getTimeBoundReference().isRewardBound())){
                std::stringstream ss;
                ss << "Unexpected type of sub-formula: " << formula.getSubformula();
                throw std::runtime_error(ss.str());
            }
        } else {
            if (!(formula.getSubformula().isTotalRewardFormula() || formula.getSubformula().isLongRunAverageRewardFormula())){
                std::stringstream ss;
                ss << "Unexpected type of sub-formula: " << formula.getSubformula();
                throw std::runtime_error(ss.str());
            }
        }
        typename SparseModelType::RewardModelType const& rewModel = model.getRewardModel(formula.asRewardOperatorFormula().getRewardModelName());

        if (rewModel.hasTransitionRewards()) {
            throw std::runtime_error("Reward model has transition rewards which is not expected.");
        }
        this->actionRewards[objIndex] = rewModel.getTotalRewardVector(model.getTransitionMatrix());
    }
}

template<typename SparseModelType>
void MultiObjectiveModel<SparseModelType>::computeSchedulerFinitelyOften(storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitionMatrix,
                                                                            storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& backwardTransitions, storm::storage::BitVector const& finitelyOftenChoices,
                                                                            storm::storage::BitVector safeStates, std::vector<uint64_t>& choices) {
    auto badStates = transitionMatrix.getRowGroupFilter(finitelyOftenChoices, true) & ~safeStates;
    // badStates shall only be reached finitely often

    auto reachBadWithProbGreater0AStates = storm::utility::graph::performProbGreater0A(
            transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardTransitions, ~safeStates, badStates, false, 0, ~finitelyOftenChoices);
    // States in ~reachBadWithProbGreater0AStates can avoid bad states forever by only taking ~finitelyOftenChoices.
    // We compute a scheduler for these states achieving exactly this (but we exclude the safe states)
    auto avoidBadStates = ~reachBadWithProbGreater0AStates & ~safeStates;
    computeSchedulerProb0(transitionMatrix, backwardTransitions, avoidBadStates, reachBadWithProbGreater0AStates, ~finitelyOftenChoices, choices);

    // We need to take care of states that will reach a bad state with prob greater 0 (including the bad states themselves).
    // due to the precondition, we know that it has to be possible to eventually avoid the bad states for ever.
    // Perform a backwards search from the avoid states and store choices with prob. 1
    computeSchedulerProb1(transitionMatrix, backwardTransitions, reachBadWithProbGreater0AStates, avoidBadStates | safeStates, choices);
}

template<class SparseModelType>
void MultiObjectiveModel<SparseModelType>::updateEcQuotient(std::vector<typename SparseModelType::ValueType> const& weightedRewardVector) {
    // Check whether we need to update the currently cached ecElimResult
    storm::storage::BitVector newTotalReward0Choices = storm::utility::vector::filterZero(weightedRewardVector);
    storm::storage::BitVector zeroLraRewardChoices(weightedRewardVector.size(), true);
    if (lraMecDecomposition) {
        throw std::runtime_error("LRA not considered.");
    }
    storm::storage::BitVector newReward0Choices = newTotalReward0Choices & zeroLraRewardChoices;
    if (!ecQuotient || ecQuotient->origReward0Choices != newReward0Choices) {
        // It is sufficient to consider the states from which a transition with non-zero reward is reachable. (The remaining states always have reward zero).
        auto nonZeroRewardStates = transitionMatrix.getRowGroupFilter(newReward0Choices, true);
        nonZeroRewardStates.complement();
        storm::storage::BitVector subsystemStates = storm::utility::graph::performProbGreater0E(
                transitionMatrix.transpose(true), storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true), nonZeroRewardStates);

        // Remove neutral end components, i.e., ECs in which no total reward is earned.
        // Note that such ECs contain one (or maybe more) LRA ECs.
        auto ecElimResult = storm::transformer::EndComponentEliminator<typename SparseModelType::ValueType>::transform(transitionMatrix, subsystemStates,
                                                                                                                       ecChoicesHint & newTotalReward0Choices, totalReward0EStates);

        storm::storage::BitVector rowsWithSumLessOne(ecElimResult.matrix.getRowCount(), false);
        for (uint64_t row = 0; row < rowsWithSumLessOne.size(); ++row) {
            if (ecElimResult.matrix.getRow(row).getNumberOfEntries() == 0) {
                rowsWithSumLessOne.set(row, true);
            } else {
                for (auto const& entry : transitionMatrix.getRow(ecElimResult.newToOldRowMapping[row])) {
                    if (!subsystemStates.get(entry.getColumn())) {
                        rowsWithSumLessOne.set(row, true);
                        break;
                    }
                }
            }
        }

        ecQuotient = EcQuotient();
        ecQuotient->matrix = std::move(ecElimResult.matrix);
        ecQuotient->ecqToOriginalChoiceMapping = std::move(ecElimResult.newToOldRowMapping);
        ecQuotient->originalToEcqStateMapping = std::move(ecElimResult.oldToNewStateMapping);
        ecQuotient->ecqToOriginalStateMapping.resize(ecQuotient->matrix.getRowGroupCount());
        for (uint64_t state = 0; state < ecQuotient->originalToEcqStateMapping.size(); ++state) {
            uint64_t ecqState = ecQuotient->originalToEcqStateMapping[state];
            if (ecqState < ecQuotient->matrix.getRowGroupCount()) {
                ecQuotient->ecqToOriginalStateMapping[ecqState].insert(state);
            }
        }
        ecQuotient->ecqStayInEcChoices = std::move(ecElimResult.sinkRows);
        ecQuotient->origReward0Choices = std::move(newReward0Choices);
        ecQuotient->origTotalReward0Choices = std::move(newTotalReward0Choices);
        ecQuotient->rowsWithSumLessOne = std::move(rowsWithSumLessOne);
        ecQuotient->auxStateValues.resize(ecQuotient->matrix.getRowGroupCount());
        ecQuotient->auxChoiceValues.resize(ecQuotient->matrix.getRowCount());
    }
}

template<class SparseModelType>
boost::optional<typename SparseModelType::ValueType> MultiObjectiveModel<SparseModelType>::computeWeightedResultBound(
        bool lower, std::vector<typename SparseModelType::ValueType> const& weightVector, storm::storage::BitVector const& objectiveFilter) const {
    auto result = storm::utility::zero<typename SparseModelType::ValueType>();
    for (auto objIndex : objectiveFilter) {
        boost::optional<typename SparseModelType::ValueType> const& objBound =
                (lower == storm::solver::minimize(this->objectives[objIndex].formula->getOptimalityType()))
                ? this->objectives[objIndex].upperResultBound : this->objectives[objIndex].lowerResultBound;
        if (objBound) {
            if (storm::solver::minimize(this->objectives[objIndex].formula->getOptimalityType())) {
                result -= objBound.get() * weightVector[objIndex];
            } else {
                result += objBound.get() * weightVector[objIndex];
            }
        } else {
            // If there is an objective without the corresponding bound we can not give guarantees for the weighted sum
            return boost::none;
        }
    }
    return result;
}


template<class SparseModelType>
void MultiObjectiveModel<SparseModelType>::setBoundsToSolver(storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver, bool requiresLower,
                                                                bool requiresUpper, uint64_t objIndex,
                                                                storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions,
                                                                storm::storage::BitVector const& rowsWithSumLessOne,
                                                                std::vector<typename SparseModelType::ValueType> const& rewards) const {
    // Check whether bounds are already available
    if (this->objectives[objIndex].lowerResultBound) {
        solver.setLowerBound(this->objectives[objIndex].lowerResultBound.get());
    }
    if (this->objectives[objIndex].upperResultBound) {
        solver.setUpperBound(this->objectives[objIndex].upperResultBound.get());
    }

    if ((requiresLower && !solver.hasLowerBound()) || (requiresUpper && !solver.hasUpperBound())) {
        computeAndSetBoundsToSolver(solver, requiresLower, requiresUpper, transitions, rowsWithSumLessOne, rewards);
    }
}

template<class SparseModelType>
void MultiObjectiveModel<SparseModelType>::setBoundsToSolver(
        storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver,
        bool requiresLower,
        bool requiresUpper,
        std::vector<typename SparseModelType::ValueType> const& weightVector,
        storm::storage::BitVector const& objectiveFilter,
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions,
        storm::storage::BitVector const& rowsWithSumLessOne,
        std::vector<typename SparseModelType::ValueType> const& rewards) {
    // Check whether bounds are already available
    boost::optional<typename SparseModelType::ValueType> lowerBound = this->computeWeightedResultBound(true, weightVector, objectiveFilter & ~lraObjectives);
    if (lowerBound) {
        if (!lraObjectives.empty()) {
            auto min = std::min_element(lraMecDecomposition->auxMecValues.begin(), lraMecDecomposition->auxMecValues.end());
            if (min != lraMecDecomposition->auxMecValues.end()) {
                lowerBound.get() += *min;
            }
        }
        solver.setLowerBound(lowerBound.get());
    }
    boost::optional<typename SparseModelType::ValueType> upperBound = this->computeWeightedResultBound(false, weightVector, objectiveFilter);
    if (upperBound) {
        if (!lraObjectives.empty()) {
            auto max = std::max_element(lraMecDecomposition->auxMecValues.begin(), lraMecDecomposition->auxMecValues.end());
            if (max != lraMecDecomposition->auxMecValues.end()) {
                upperBound.get() += *max;
            }
        }
        solver.setUpperBound(upperBound.get());
    }

    if ((requiresLower && !solver.hasLowerBound()) || (requiresUpper && !solver.hasUpperBound())) {
        computeAndSetBoundsToSolver(solver, requiresLower, requiresUpper, transitions, rowsWithSumLessOne, rewards);
    }
}

template<class SparseModelType>
void MultiObjectiveModel<SparseModelType>::computeAndSetBoundsToSolver(storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver,
                                                                          bool requiresLower,
                                                                          bool requiresUpper,
                                                                          storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitions,
                                                                          storm::storage::BitVector const& rowsWithSumLessOne,
                                                                          std::vector<typename SparseModelType::ValueType> const& rewards) const {
    // Compute the one step target probs
    std::vector<typename SparseModelType::ValueType> oneStepTargetProbs(transitions.getRowCount(), storm::utility::zero<typename SparseModelType::ValueType>());
    for (auto row : rowsWithSumLessOne) {
        oneStepTargetProbs[row] = storm::utility::one<typename SparseModelType::ValueType>() - transitions.getRowSum(row);
    }

    if (requiresLower && !solver.hasLowerBound()) {
        // Compute lower bounds
        std::vector<typename SparseModelType::ValueType> negativeRewards;
        negativeRewards.reserve(transitions.getRowCount());
        uint64_t row = 0;
        for (auto const& rew : rewards) {
            if (rew < storm::utility::zero<typename SparseModelType::ValueType>()) {
                negativeRewards.resize(row, storm::utility::zero<typename SparseModelType::ValueType>());
                negativeRewards.push_back(-rew);
            }
            ++row;
        }
        if (!negativeRewards.empty()) {
            negativeRewards.resize(row, storm::utility::zero<typename SparseModelType::ValueType>());
            std::vector<typename SparseModelType::ValueType> lowerBounds =
                    storm::modelchecker::helper::DsMpiMdpUpperRewardBoundsComputer<typename SparseModelType::ValueType>(transitions, negativeRewards, oneStepTargetProbs)
                            .computeUpperBounds();
            storm::utility::vector::scaleVectorInPlace(lowerBounds, -storm::utility::one<typename SparseModelType::ValueType>());
            solver.setLowerBounds(std::move(lowerBounds));
        } else {
            solver.setLowerBound(storm::utility::zero<typename SparseModelType::ValueType>());
        }
    }

    // Compute upper bounds
    if (requiresUpper && !solver.hasUpperBound()) {
        std::vector<typename SparseModelType::ValueType> positiveRewards;
        positiveRewards.reserve(transitions.getRowCount());
        uint64_t row = 0;
        for (auto const& rew : rewards) {
            if (rew > storm::utility::zero<typename SparseModelType::ValueType>()) {
                positiveRewards.resize(row, storm::utility::zero<typename SparseModelType::ValueType>());
                positiveRewards.push_back(rew);
            }
            ++row;
        }
        if (!positiveRewards.empty()) {
            positiveRewards.resize(row, storm::utility::zero<typename SparseModelType::ValueType>());
            solver.setUpperBound(
                    storm::modelchecker::helper::BaierUpperRewardBoundsComputer<typename SparseModelType::ValueType>(transitions, positiveRewards, oneStepTargetProbs).computeUpperBound());
        } else {
            solver.setUpperBound(storm::utility::zero<typename SparseModelType::ValueType>());
        }
    }
}

template<typename SparseModelType>
std::vector<uint64_t> MultiObjectiveModel<SparseModelType>::computeValidInitialScheduler(
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> &matrix,
        storm::storage::BitVector &rowsWithSumLessOne) {

    std::vector<uint64_t> result(matrix.getRowGroupCount());
    auto const& groups = matrix.getRowGroupIndices();
    auto backwardsTransitions = matrix.transpose(true);
    storm::storage::BitVector processedStates(result.size(), false);
    for (uint64_t state = 0; state < result.size(); ++state) {
        if (rowsWithSumLessOne.getNextSetIndex(groups[state]) < groups[state + 1]) {
            result[state] = rowsWithSumLessOne.getNextSetIndex(groups[state]) - groups[state];
            processedStates.set(state, true);
        }
    }

    computeSchedulerProb1(matrix, backwardsTransitions, ~processedStates, processedStates, result);
    return result;
}


template<typename SparseModelType>
void MultiObjectiveModel<SparseModelType>::computeSchedulerProb1(
        const storm::storage::SparseMatrix<typename SparseModelType::ValueType> &transitionMatrix,
        const storm::storage::SparseMatrix<typename SparseModelType::ValueType> &backwardTransitions,
        const storm::storage::BitVector &consideredStates, const storm::storage::BitVector &statesToReach,
        std::vector<uint64_t> &choices, const storm::storage::BitVector *allowedChoices) const {
    std::vector<uint64_t> stack;
    storm::storage::BitVector processedStates = statesToReach;
    stack.insert(stack.end(), processedStates.begin(), processedStates.end());
    uint64_t currentState = 0;

    while (!stack.empty()) {
        currentState = stack.back();
        stack.pop_back();

        for (auto const& predecessorEntry : backwardTransitions.getRow(currentState)) {
            auto predecessor = predecessorEntry.getColumn();
            if (consideredStates.get(predecessor) & !processedStates.get(predecessor)) {
                // Find a choice leading to an already processed state (such a choice has to exist since this is a predecessor of the currentState)
                auto const& groupStart = transitionMatrix.getRowGroupIndices()[predecessor];
                auto const& groupEnd = transitionMatrix.getRowGroupIndices()[predecessor + 1];
                uint64_t row = allowedChoices ? allowedChoices->getNextSetIndex(groupStart) : groupStart;
                for (; row < groupEnd; row = allowedChoices ? allowedChoices->getNextSetIndex(row + 1) : row + 1) {
                    bool hasSuccessorInProcessedStates = false;
                    for (auto const& successorOfPredecessor : transitionMatrix.getRow(row)) {
                        if (processedStates.get(successorOfPredecessor.getColumn())) {
                            hasSuccessorInProcessedStates = true;
                            break;
                        }
                    }
                    if (hasSuccessorInProcessedStates) {
                        choices[predecessor] = row - groupStart;
                        processedStates.set(predecessor, true);
                        stack.push_back(predecessor);
                        break;
                    }
                }
                if (!(allowedChoices || row < groupEnd)){
                    std::cout << "Unable to find choice at a predecessor of a processed state that leads to a processed state.\n";
                }
            }
        }
    }
    if(!consideredStates.isSubsetOf(processedStates)){
        std::cout << "Not all states have been processed.\n";
    }

}

template<typename SparseModelType>
void MultiObjectiveModel<SparseModelType>::computeSchedulerProb0(
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& transitionMatrix,
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> const& backwardTransitions,
        storm::storage::BitVector const& consideredStates,
        storm::storage::BitVector const& statesToAvoid,
        storm::storage::BitVector const& allowedChoices, std::vector<uint64_t>& choices) const {
    for (auto state : consideredStates) {
        auto const& groupStart = transitionMatrix.getRowGroupIndices()[state];
        auto const& groupEnd = transitionMatrix.getRowGroupIndices()[state + 1];
        bool choiceFound = false;
        for (uint64_t row = allowedChoices.getNextSetIndex(groupStart); row < groupEnd; row = allowedChoices.getNextSetIndex(row + 1)) {
            choiceFound = true;
            for (auto const& element : transitionMatrix.getRow(row)) {
                if (statesToAvoid.get(element.getColumn())) {
                    choiceFound = false;
                    break;
                }
            }
            if (choiceFound) {
                choices[state] = row - groupStart;
                break;
            }
        }
        if (!choiceFound) {
            std::cout << "Unable to find choice for a state.\n";
        }
    }
}

template<class SparseModelType>
void MultiObjectiveModel<SparseModelType>::transformEcqSolutionToOriginalModel(std::vector<typename SparseModelType::ValueType> const& ecqSolution,
                                                                                  std::vector<uint_fast64_t> const& ecqOptimalChoices,
                                                                                  std::map<uint64_t, uint64_t> const& ecqStateToOptimalMecMap,
                                                                                  std::vector<typename SparseModelType::ValueType>& originalSolution,
                                                                                  std::vector<uint_fast64_t>& originalOptimalChoices) const {
    auto backwardsTransitions = transitionMatrix.transpose(true);

    // Keep track of states for which no choice has been set yet.
    storm::storage::BitVector unprocessedStates(transitionMatrix.getRowGroupCount(), true);

    // For each eliminated ec, keep track of the states (within the ec) that we want to reach and the states for which a choice needs to be set
    // (Declared already at this point to avoid expensive allocations in each loop iteration)
    storm::storage::BitVector ecStatesToReach(transitionMatrix.getRowGroupCount(), false);
    storm::storage::BitVector ecStatesToProcess(transitionMatrix.getRowGroupCount(), false);

    // Run through each state of the ec quotient as well as the associated state(s) of the original model
    for (uint64_t ecqState = 0; ecqState < ecqSolution.size(); ++ecqState) {
        uint64_t ecqChoice = ecQuotient->matrix.getRowGroupIndices()[ecqState] + ecqOptimalChoices[ecqState];
        uint_fast64_t origChoice = ecQuotient->ecqToOriginalChoiceMapping[ecqChoice];
        auto const& origStates = ecQuotient->ecqToOriginalStateMapping[ecqState];
        if(origStates.empty()) {
            std::cout << "Unexpected empty set of original states.\n";
        }
        if (ecQuotient->ecqStayInEcChoices.get(ecqChoice)) {
            // We stay in the current state(s) forever (End component)
            // We need to set choices in a way that (i) the optimal LRA Mec is reached (if there is any) and (ii) 0 total reward is collected.
            if (!ecqStateToOptimalMecMap.empty()) {
                throw std::runtime_error("This framework does not deal with LRA");
            } else {
                // If there is no LRA Mec to reach, we just need to make sure that finite total reward is collected for all objectives
                // In this branch our BitVectors have a slightly different meaning, so we create more readable aliases
                storm::storage::BitVector& ecStatesToAvoid = ecStatesToReach;
                bool needSchedulerComputation = false;
                if (!storm::utility::isZero(ecqSolution[ecqState])){
                    std::stringstream ss;
                    ss << "Solution for state that stays inside EC must be zero. Got " << ecqSolution[ecqState] << " instead.";
                    throw std::runtime_error(ss.str());
                }
                for (auto const& state : origStates) {
                    originalSolution[state] = storm::utility::zero<typename SparseModelType::ValueType>();  // i.e. ecqSolution[ecqState];
                    ecStatesToProcess.set(state, true);
                }
                auto validChoices = transitionMatrix.getRowFilter(ecStatesToProcess, ecStatesToProcess);
                auto valid0RewardChoices = validChoices & actionsWithoutRewardInUnboundedPhase;
                for (auto const& state : origStates) {
                    auto groupStart = transitionMatrix.getRowGroupIndices()[state];
                    auto groupEnd = transitionMatrix.getRowGroupIndices()[state + 1];
                    auto nextValidChoice = valid0RewardChoices.getNextSetIndex(groupStart);
                    if (nextValidChoice < groupEnd) {
                        originalOptimalChoices[state] = nextValidChoice - groupStart;
                    } else {
                        // this state should not be reached infinitely often
                        ecStatesToAvoid.set(state, true);
                        needSchedulerComputation = true;
                    }
                }
                if (needSchedulerComputation) {
                    // There are ec states which we should not visit infinitely often
                    auto ecStatesThatCanAvoid =
                            storm::utility::graph::performProbGreater0A(transitionMatrix, transitionMatrix.getRowGroupIndices(), backwardsTransitions,
                                                                        ecStatesToProcess, ecStatesToAvoid, false, 0, valid0RewardChoices);
                    ecStatesThatCanAvoid.complement();
                    // Set the choice for all states that can achieve value 0
                    computeSchedulerProb0(transitionMatrix, backwardsTransitions, ecStatesThatCanAvoid, ecStatesToAvoid, valid0RewardChoices,
                                          originalOptimalChoices);
                    // Set the choice for all remaining states
                    computeSchedulerProb1(transitionMatrix, backwardsTransitions, ecStatesToProcess & ~ecStatesToAvoid, ecStatesToAvoid, originalOptimalChoices,
                                          &validChoices);
                }
                ecStatesToAvoid.clear();
                ecStatesToProcess.clear();
            }
        } else {
            // We eventually leave the current state(s)
            // In this case, we can safely take the origChoice at the corresponding state (say 's').
            // For all other origStates associated with ecqState (if there are any), we make sure that the state 's' is reached almost surely.
            if (origStates.size() > 1) {
                for (auto const& state : origStates) {
                    // Check if the orig choice originates from this state
                    auto groupStart = transitionMatrix.getRowGroupIndices()[state];
                    auto groupEnd = transitionMatrix.getRowGroupIndices()[state + 1];
                    if (origChoice >= groupStart && origChoice < groupEnd) {
                        originalOptimalChoices[state] = origChoice - groupStart;
                        ecStatesToReach.set(state, true);
                    } else {
                        if(!(origStates.size() > 1)){
                            std::cout << "Multiple original states expected.\n";
                        }
                        ecStatesToProcess.set(state, true);
                    }
                    unprocessedStates.set(state, false);
                    originalSolution[state] = ecqSolution[ecqState];
                }
                computeSchedulerProb1(transitionMatrix, backwardsTransitions, ecStatesToProcess, ecStatesToReach, originalOptimalChoices,
                                      &ecQuotient->origTotalReward0Choices);
                // Clear bitvectors for next ecqState.
                ecStatesToProcess.clear();
                ecStatesToReach.clear();
            } else {
                // There is just one state so we take the associated choice.
                auto state = *origStates.begin();
                auto groupStart = transitionMatrix.getRowGroupIndices()[state];
                if(!(origChoice >= groupStart && origChoice < transitionMatrix.getRowGroupIndices()[state + 1])) {
                    std::cout << "Invalid choice: " << originalOptimalChoices[state] << " at a state with " <<
                              transitionMatrix.getRowGroupSize(state) << " choices.\n";
                }
                originalOptimalChoices[state] = origChoice - groupStart;
                originalSolution[state] = ecqSolution[ecqState];
                unprocessedStates.set(state, false);
            }
        }
    }


    // The states that still not have been processed, there is no associated state of the ec quotient.
    // This is because the value for these states will be 0 under all (lra optimal-) schedulers.
    storm::utility::vector::setVectorValues(
            originalSolution, unprocessedStates, storm::utility::zero<typename SparseModelType::ValueType>());
    // Get a set of states for which we know that no reward (for all objectives) will be collected
    if (this->lraMecDecomposition) {
        // In this case, all unprocessed non-lra mec states should reach an (unprocessed) lra mec
        for (auto const& mec : this->lraMecDecomposition->mecs) {
            for (auto const& sc : mec) {
                if (unprocessedStates.get(sc.first)) {
                    ecStatesToReach.set(sc.first, true);
                }
            }
        }
    } else {
        ecStatesToReach = unprocessedStates & totalReward0EStates;
        // Set a scheduler for the ecStates that we want to reach
        computeSchedulerProb0(transitionMatrix, backwardsTransitions, ecStatesToReach, ~unprocessedStates | ~totalReward0EStates,
                              actionsWithoutRewardInUnboundedPhase, originalOptimalChoices);
    }
    unprocessedStates &= ~ecStatesToReach;
    // Set a scheduler for the remaining states
    computeSchedulerProb1(transitionMatrix, backwardsTransitions, unprocessedStates, ecStatesToReach, originalOptimalChoices);
}

// Explicit

template void mopmc::multiobjective::performMultiObjectiveModelChecking(
    storm::Environment env, storm::models::sparse::Mdp<double> &model,
    const storm::logic::MultiObjectiveFormula &formula);

template class MultiObjectiveModel<storm::models::sparse::Mdp<double>>;

}
}
