//
// Created by thomas on 6/09/23.
//
#include <storm/models/sparse/Mdp.h>
#include "StandardMdpPcaaChecker.h"
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include <storm/transformer/EndComponentEliminator.h>
#include <storm/transformer/GoalStateMerger.h>
#include <storm/utility/vector.h>
#include <set>
#include <storm/solver/MinMaxLinearEquationSolver.h>
#include "../solvers/InducedEquationSolver.h"
#include "../solvers/IterativeSolver.h"
#include <storm/modelchecker/prctl/helper/DsMpiUpperRewardBoundsComputer.h>
#include <storm/modelchecker/prctl/helper/BaierUpperRewardBoundsComputer.h>

namespace mopmc{

namespace multiobjective{

template<typename SparseModelType>
StandardMdpPcaaChecker<SparseModelType>::StandardMdpPcaaChecker(
    storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<SparseModelType> &preprocessorResult)
    : objectives(preprocessorResult.objectives){
    initialise(preprocessorResult);
}

template<typename SparseModelType>
void StandardMdpPcaaChecker<SparseModelType>::initialise(
    storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessorResult<SparseModelType> &preprocessorResult) {
    auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<SparseModelType>::analyze(preprocessorResult);
    if(rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::Infinite){
        throw std::runtime_error("There is no Pareto optimal scheduler that yields finite reward for all objectives. This is not supported.");
    }
    if (!rewardAnalysis.totalRewardLessInfinityEStates){
        throw std::runtime_error("The set of states with reward < infinity for some scheduler has not been computed during preprocessing.");
    }
    if(!preprocessorResult.containsOnlyTrivialObjectives()){
        throw std::runtime_error("At least one objective was not reduced to an expected (long run, total or cumulative) reward objective during preprocessing. This is not"
                                 "supported by the considered weight vector checker.");
    }
    if(preprocessorResult.preprocessedModel->getInitialStates().getNumberOfSetBits() != 1){
        throw std::runtime_error("The model has multiple initial states.");
    }

    // build a subsystem of the preprocessor result model that discards states that yield infinite
    // reward for all schedulers.
    // We can also merge the states that will have reward zero anyway
    storm::storage::BitVector maybeStates = rewardAnalysis.totalRewardLessInfinityEStates.get() & ~rewardAnalysis.reward0AStates;
    storm::storage::BitVector finiteTotalRewardChoices = preprocessorResult.preprocessedModel->getTransitionMatrix().getRowFilter(
            rewardAnalysis.totalRewardLessInfinityEStates.get(), rewardAnalysis.totalRewardLessInfinityEStates.get());
    std::set<std::string> relevantRewardModels;
    for (auto& obj : this->objectives) {
        obj.formula->gatherReferencedRewardModels(relevantRewardModels);
    }
    storm::transformer::GoalStateMerger<SparseModelType> merger(*preprocessorResult.preprocessedModel);

    auto mergerResult = merger.mergeTargetAndSinkStates(
        maybeStates, rewardAnalysis.reward0AStates,
        storm::storage::BitVector(maybeStates.size(), false),
        std::vector<std::string>(relevantRewardModels.begin(), relevantRewardModels.end()),
        finiteTotalRewardChoices);

    // initialise data specific for the considered model type
    initialiseModelTypeSpecificData(*mergerResult.model);

    // initialise the general data of the model
    transitionMatrix = std::move(mergerResult.model->getTransitionMatrix());
    initialState = *mergerResult.model->getInitialStates().begin();
    totalReward0EStates = rewardAnalysis.totalReward0EStates % maybeStates;
    if (mergerResult.targetState) {
        // there is an additional state in the result
        std::cout << "There is an additional state in the result\n";

        totalReward0EStates.resize(totalReward0EStates.size() + 1, true);

        // The over approximation for the possible ec consists of the states that can reach the
        // target states with prob 0 and the target states
        storm::storage::BitVector targetStatesAsVector(transitionMatrix.getRowGroupCount(), false);
        targetStatesAsVector.set(*mergerResult.targetState, true);
        ecChoicesHint = transitionMatrix.getRowFilter(
            storm::utility::graph::performProb0E(transitionMatrix, transitionMatrix.getRowGroupIndices(),
                                                 transitionMatrix.transpose(true),
                                                 storm::storage::BitVector(targetStatesAsVector.size(), true), targetStatesAsVector));
        ecChoicesHint.set(transitionMatrix.getRowGroupIndices()[*mergerResult.targetState], true);
    } else {
        ecChoicesHint = storm::storage::BitVector(transitionMatrix.getRowCount(), true);
    }

    // set data for unbounded objectives
    lraObjectives = storm::storage::BitVector(this->objectives.size(), false);
    objectivesWithNoUpperTimeBound = storm::storage::BitVector(this->objectives.size(), false);
    actionsWithoutRewardInUnboundedPhase = storm::storage::BitVector(transitionMatrix.getRowCount(), true);
    for(uint_fast64_t objIndex = 0; objIndex < this->objectives.size(); ++objIndex) {
        auto& formula = *this->objectives[objIndex].formula;
        if (formula.getSubformula().isTotalRewardFormula()) {
            std::cout << "Formula " << formula << " is total rewards formula. Set as "
                                                  "objective with no upper time bound.\n";
            objectivesWithNoUpperTimeBound.set(objIndex, true);
            actionsWithoutRewardInUnboundedPhase &= storm::utility::vector::filterZero(actionRewards[objIndex]);
        }
        if (formula.getSubformula().isLongRunAverageRewardFormula()) {
            throw std::runtime_error("Only those objectives with a threshold are considered");
        }
    }

    // Print some stats of the model
    std::cout << "Final preprocessed model has " << transitionMatrix.getRowGroupCount() << " states\n";
    std::cout << "Final preprocessed model has " << transitionMatrix.getRowCount() << " actions\n";
}


template <typename SparseModelType>
void StandardMdpPcaaChecker<SparseModelType>::initialiseModelTypeSpecificData(SparseModelType &model) {
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

template <typename SparseModelType>
void StandardMdpPcaaChecker<SparseModelType>::updateEcQuotient(std::vector<typename SparseModelType::ValueType>& weightedRewardVector) {
    // checks whether we need to update the currently cached EcElimResult
    storm::storage::BitVector newReward0Choices = storm::utility::vector::filterZero(weightedRewardVector);
    // TODO: future work note:
    //  We don't consider LRA
    if (!ecQuotient || ecQuotient->origReward0Choices != newReward0Choices) {
        // It suffices to consider the states from which a transition with non-zero reward
        // is reachable. The remaining states always have zero reward.
        auto nonZeroRewardStates = transitionMatrix.getRowGroupFilter(newReward0Choices, true);
        nonZeroRewardStates.complement();
        // TODO: might need to rewrite performProbGreater0E to accept an Eigen matrix in its
        //  signature
        storm::storage::BitVector subsystemStates = storm::utility::graph::performProbGreater0E(
                transitionMatrix.transpose(true),
                storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true),
                nonZeroRewardStates);

        // remove neutral end components, i.e. ECs in which no total reward is earned
        auto ecElimResult = storm::transformer::EndComponentEliminator<typename SparseModelType::ValueType>::transform(
                transitionMatrix,
                subsystemStates,
                ecChoicesHint & newReward0Choices,
                totalReward0EStates);

        storm::storage::BitVector rowsWithSumLessOne(ecElimResult.matrix.getRowCount(), false);
        for (uint_fast64_t row = 0; row < rowsWithSumLessOne.size(); ++row) {
            if (ecElimResult.matrix.getRow(row).getNumberOfEntries() == 0) {
                rowsWithSumLessOne.set(row, true);
            } else {
                for (auto &entry: transitionMatrix.getRow(ecElimResult.newToOldRowMapping[row])) {
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
        storm::storage::BitVector copyNewReward0Choices = newReward0Choices;
        ecQuotient->origReward0Choices = std::move(newReward0Choices);
        // TODO: note that if LRAs computations are done in future work then this next line
        //  will need to change
        ecQuotient->origTotalReward0Choices = std::move(copyNewReward0Choices);
        ecQuotient->rowsWithSumLessOne = std::move(rowsWithSumLessOne);
        ecQuotient->auxStateValues.resize(ecQuotient->matrix.getRowGroupCount());
        ecQuotient->auxChoiceValues.resize(ecQuotient->matrix.getRowCount());
    }
}

template <typename SparseModelType>
void StandardMdpPcaaChecker<SparseModelType>::unboundedWeightPhase(const storm::Environment &env,
                                                                   const std::vector<typename SparseModelType::ValueType> &weightVector) {
    auto totalRewardObjectives = objectivesWithNoUpperTimeBound;
    std::vector<typename SparseModelType::ValueType> weightedRewardVector(
            transitionMatrix.getRowCount(), storm::utility::zero<typename SparseModelType::ValueType>());
    for (auto objIndex: totalRewardObjectives){
        typename SparseModelType::ValueType weight = storm::solver::minimize(this->objectives[objIndex].formula->getOptimalityType());
        storm::utility::vector::addScaledVector(weightedRewardVector, actionRewards[objIndex], weight);
    }

    updateEcQuotient(weightedRewardVector);
    // Set up the choice values
    storm::utility::vector::selectVectorValues(ecQuotient->auxChoiceValues, ecQuotient->ecqToOriginalChoiceMapping, weightedRewardVector);
    std::map<uint64_t, uint64_t> ecqStateToOptimalMecMap;
    storm::solver::GeneralMinMaxLinearEquationSolverFactory<typename SparseModelType::ValueType> solverFactory;
    std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<typename SparseModelType::ValueType>> solver = solverFactory.create(env, ecQuotient->matrix);
    solver->setTrackScheduler(true);
    solver->setHasUniqueSolution(true);
    solver->setOptimizationDirection(storm::solver::OptimizationDirection::Maximize);
    auto req = solver->getRequirements(env, storm::solver::OptimizationDirection::Maximize);
    setBoundsToSolver(*solver, req.lowerBounds(), req.upperBounds(), weightVector, objectivesWithNoUpperTimeBound, ecQuotient->matrix,
                      ecQuotient->rowsWithSumLessOne, ecQuotient->auxChoiceValues);
    if (solver->hasLowerBound()) {
        req.clearLowerBounds();
    }
    if (solver->hasUpperBound()) {
        req.clearUpperBounds();
    }
    if (req.validInitialScheduler()) {
        solver->setInitialScheduler(computeValidInitialScheduler(ecQuotient->matrix, ecQuotient->rowsWithSumLessOne));
        req.clearValidInitialScheduler();
    }
    if(req.hasEnabledCriticalRequirement()) {
        std::stringstream ss;
        ss << "Solver requirements " + req.getEnabledRequirementsAsString() + " not checked.";
        throw std::runtime_error(ss.str());
    }
    solver->setRequirementsChecked(true);

    // Use the (0...0) vector as initial guess for the solution.
    std::fill(ecQuotient->auxStateValues.begin(), ecQuotient->auxStateValues.end(), storm::utility::zero<typename SparseModelType::ValueType>());

    solver->solveEquations(env, ecQuotient->auxStateValues, ecQuotient->auxChoiceValues);
    this->weightedResult = std::vector<typename SparseModelType::ValueType>(transitionMatrix.getRowGroupCount());
    solver->getSchedulerChoices();
}

template<class SparseModelType>
boost::optional<typename SparseModelType::ValueType> StandardMdpPcaaChecker<SparseModelType>::computeWeightedResultBound(
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
void StandardMdpPcaaChecker<SparseModelType>::setBoundsToSolver(
        storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver,
        bool requiresLower,
        bool requiresUpper, std::vector<typename SparseModelType::ValueType> const& weightVector,
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
void StandardMdpPcaaChecker<SparseModelType>::computeAndSetBoundsToSolver(storm::solver::AbstractEquationSolver<typename SparseModelType::ValueType>& solver,
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

template <typename SparseModelType>
void StandardMdpPcaaChecker<SparseModelType>::check(std::vector<typename SparseModelType::ValueType> &w){
    Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> wEig(w.data(), w.size());
    // do one multiplication of the rewards matrix and the weight vector
    auto totalRewardObjectives = objectivesWithNoUpperTimeBound;
    std::vector<typename SparseModelType::ValueType> weightedRewardVector(
            transitionMatrix.getRowCount(), storm::utility::zero<typename SparseModelType::ValueType>());
    for (auto objIndex: totalRewardObjectives){
        typename SparseModelType::ValueType weight = storm::solver::minimize(this->objectives[objIndex].formula->getOptimalityType());
        storm::utility::vector::addScaledVector(weightedRewardVector, actionRewards[objIndex], weight);
    }

    updateEcQuotient(weightedRewardVector);

    // set up the choice values
    storm::utility::vector::selectVectorValues(ecQuotient->auxChoiceValues, ecQuotient->ecqToOriginalChoiceMapping, weightedRewardVector);
    // TODO if lra is used in future work it needs to be added here
    // convert the transition matrix to the Eigen form
    std::vector<uint64_t> initSch = computeValidInitialScheduler(ecQuotient->matrix, ecQuotient->rowsWithSumLessOne);
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> identity = makeEigenIdentityMatrix();

    toEigenSparseMatrix(); // The transition matrix is now an Eigen matrix
    // compute the value of the initial policy, which is an induced DTMC
    Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> x(ecQuotient->auxStateValues.data(), ecQuotient->auxStateValues.size());

    mopmc::solver::iter::policyIteration<SparseModelType>(
            eigenTransitionMatrix,
            identity,
            x,
            ecQuotient->auxChoiceValues,
            initSch,
            ecQuotient->matrix.getRowGroupIndices());

};

template<typename SparseModelType>
void StandardMdpPcaaChecker<SparseModelType>::toEigenSparseMatrix() {
    std::vector<Eigen::Triplet<typename SparseModelType::ValueType>> triplets;
    triplets.reserve(ecQuotient->matrix.getNonzeroEntryCount());

    for(uint_fast64_t row = 0; row < ecQuotient->matrix.getRowCount(); ++row) {
        for(auto element : ecQuotient->matrix.getRow(row)) {
            triplets.emplace_back(row, element.getColumn(), element.getValue());
        }
    }

    Eigen::SparseMatrix<typename SparseModelType::ValueType,  Eigen::RowMajor> result =
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>(
                ecQuotient->matrix.getRowCount(), ecQuotient->matrix.getColumnCount()
        );
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    this->eigenTransitionMatrix = result;
}

template<typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> StandardMdpPcaaChecker<SparseModelType>::makeEigenIdentityMatrix() {
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> I (ecQuotient->matrix.getRowGroupCount(), ecQuotient->matrix.getRowGroupCount());
    for (uint_fast64_t i = 0; i < ecQuotient->matrix.getRowGroupCount(); ++i) {
        I.insert(i, i) = static_cast<typename SparseModelType::ValueType>(1.0);
    }
    I.finalize();
    return I;
}

template<typename SparseModelType>
std::vector<uint64_t> StandardMdpPcaaChecker<SparseModelType>::computeValidInitialScheduler(
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
void StandardMdpPcaaChecker<SparseModelType>::computeSchedulerProb1(
    const storm::storage::SparseMatrix<typename SparseModelType::ValueType> &transitionMatrix,
    const storm::storage::SparseMatrix<typename SparseModelType::ValueType> &backwardTransitions,
    const storm::storage::BitVector &consideredStates, const storm::storage::BitVector &statesToReach,
    std::vector<uint64_t> &choices, const storm::storage::BitVector *allowedChoices) {
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

template class StandardMdpPcaaChecker<storm::models::sparse::Mdp<double>>;

}
}
