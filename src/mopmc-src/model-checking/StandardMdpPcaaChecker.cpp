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
#include "../solvers/InducedEquationSolver.h"
#include "../solvers/ValueIteration.h"

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
    Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> b(ecQuotient->auxStateValues.data(), ecQuotient->auxStateValues.size());
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> dtmc = eigenInducedTransitionMatrix(
        eigenTransitionMatrix, initSch);

    mopmc::solver::linsystem::solverHelper(b, x, dtmc, identity);


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

template <typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> StandardMdpPcaaChecker<SparseModelType>::eigenInducedTransitionMatrix(
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>& fullTransitionSystem,
    std::vector<uint64_t>& chosenActions) {

    assert(chosenActions.size() == ecQuotient->matrix.getColumnCount());
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> subMatrix(chosenActions.size(), chosenActions.size());
    for(uint_fast64_t state = 0; state < ecQuotient->matrix.getColumnCount(); ++state) {
        typename Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>::InnerIterator it(fullTransitionSystem, chosenActions[state]);
        for (; it; ++it) {
            subMatrix.insert(state, it.col()) = it.value();
        }
    }

    subMatrix.makeCompressed();
    return subMatrix;
};

template class StandardMdpPcaaChecker<storm::models::sparse::Mdp<double>>;

}
}
