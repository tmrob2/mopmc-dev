//
// Created by thomas on 18/08/23.
//
#include "MultiObjective.h"
#include <storm/storage/BitVector.h>
#include <storm/logic/OperatorFormula.h>
#include <storm/modelchecker/results/ExplicitQualitativeCheckResult.h>
#include <storm/modelchecker/results/ExplicitQuantitativeCheckResult.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/utility/vector.h>
#include <storm/utility/FilteredRewardModel.h>

#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/modelchecker/prctl/helper/BaierUpperRewardBoundsComputer.h>
#include <storm/settings/SettingsManager.h>
#include <storm/storage/expressions/ExpressionManager.h>
#include <storm/transformer/MemoryIncorporation.h>
#include <storm/utility/macros.h>

#include <storm/exceptions/InvalidPropertyException.h>
#include <storm/logic/RewardAccumulation.h>

template<typename T>
void mopmc::multiobj::performMultiObjectiveModelChecking(mopmc::sparse::SparseModelBuilder<T> &spModel,
                                                         const storm::logic::MultiObjectiveFormula &formula) {
    mopmc::multiobj::preprocess(spModel, formula);
}

template <typename T>
void mopmc::multiobj::preprocess(mopmc::sparse::SparseModelBuilder<T> &spModel,
                                 const storm::logic::MultiObjectiveFormula &formula) {
    boost::optional<std::string> deadlockLabel;
    mopmc::multiobj::reduceStates(spModel, deadlockLabel, formula);
}

template<typename T>
void mopmc::multiobj::reduceStates(mopmc::sparse::SparseModelBuilder<T> &spModel,
                                   boost::optional<std::string> deadlockLabel,
                                   storm::logic::MultiObjectiveFormula const& formula) {
    // Assume that every state is important
    storm::storage::BitVector absorbingStates(spModel.getNumberOfStates(), true);
    auto mopmcModel = mopmc::model_checking::DTMCModelSolver<double>(spModel);

    for (auto const& opFormula : formula.getSubformulas()) {
        storm::storage::BitVector absorbingStatesForSubFormula;
        storm::storage::BitVector stormAbsorbingStatesForSubFormula;
        if (!opFormula->isOperatorFormula()) {
            std::cout << "Cannot process this formula\n";
        }

        auto const& pathFormula = opFormula->asOperatorFormula().getSubformula();
        if(opFormula ->isProbabilityOperatorFormula()) {
            if (pathFormula.isUntilFormula()) {
                std::cout << "Path formula is until formula\n";
                auto lhs = mopmcModel.check(pathFormula.asUntilFormula().getLeftSubformula())
                        -> asExplicitQualitativeCheckResult().getTruthValuesVector();
                auto rhs = mopmcModel.check(pathFormula.asUntilFormula().getRightSubformula())
                        -> asExplicitQualitativeCheckResult().getTruthValuesVector();

                absorbingStatesForSubFormula = mopmc::graph::performProb0A<double>(
                        spModel,lhs,rhs);
                absorbingStatesForSubFormula |= mopmc::multiobj::getOnlyReachableViaPhi(spModel, ~lhs | rhs);
            } else if (pathFormula.isBoundedUntilFormula()) {
                std::cout<< "Bounded until formula\n";
            } else if (pathFormula.isGloballyFormula()) {
                std::cout << "Is global formula\n";

                auto phi = mopmcModel.check(pathFormula.asGloballyFormula().getSubformula())->
                        asExplicitQualitativeCheckResult().getTruthValuesVector();
                auto notPhi = ~phi;
                absorbingStatesForSubFormula = mopmc::graph::performProb0A<double>(
                        spModel,phi,notPhi);
                if (stormAbsorbingStatesForSubFormula != absorbingStatesForSubFormula) {
                    std::cout << "The Globally subformula check produced the same bitVector\n";
                } else {
                    std::cout << "The Globally subformula check did not produce the same bitVector\n";
                }
                std::cout << "Finished backward check: globally\n";
                std::cout << "Absorbing states: " << absorbingStatesForSubFormula.getNumberOfSetBits() << "\n";
                absorbingStatesForSubFormula |= mopmc::multiobj::getOnlyReachableViaPhi(spModel, notPhi);
            } else if (pathFormula.isEventuallyFormula()) {
                std::cout << "Is eventual formula\n";

                auto phi = mopmcModel.check(pathFormula.asEventuallyFormula().getSubformula()) ->
                        asExplicitQualitativeCheckResult().getTruthValuesVector();
                absorbingStatesForSubFormula = mopmc::graph::performProb0A<double>(
                        spModel,~phi,phi);
                std::cout << "Finished backward check: eventually\n";
                absorbingStatesForSubFormula |= mopmc::multiobj::getOnlyReachableViaPhi(spModel, phi);
                std::cout << "Number of reachable states remaining: " << absorbingStatesForSubFormula.getNumberOfSetBits() << "\n";
            } else {
                throw std::runtime_error("Probability formula type not implemented");
            }

        } else if (opFormula -> isRewardOperatorFormula()) {

            auto const& baseRewardModel = spModel.getRewardModel(opFormula->asRewardOperatorFormula().getRewardModelName());
            if (pathFormula.isEventuallyFormula()){
                auto const& rewardModel = storm::utility::createFilteredRewardModel(baseRewardModel, false, pathFormula.asEventuallyFormula());

                std::function<bool(T const&)> const& filter = storm::utility::isZero<T>;

                storm::storage::BitVector statesWithZeroReward = rewardModel.get().hasStateRewards() ?
                                                                 storm::utility::vector::filter(rewardModel.get().getStateRewardVector(), filter) :
                                                                 storm::storage::BitVector(spModel.getNumberOfStates(), true);

                if (rewardModel.get().hasStateActionRewards()) {
                    std::cout << "has state action rewards\n";
                    for(uint_fast64_t state = 0; state < spModel.getNumberOfStates(); ++state) {
                        std::vector<uint_fast64_t> rows = spModel.getActionsForState(state);
                        for (uint_fast64_t row: rows) {
                            if (!filter(rewardModel.get().getStateActionRewardVector()[row])) {
                                statesWithZeroReward.set(state, false);
                                break;
                            }
                        }
                    }
                }

                absorbingStatesForSubFormula = mopmc::graph::performProb0A<double>(
                        spModel, statesWithZeroReward, ~statesWithZeroReward);
                auto phi = mopmcModel.check(pathFormula.asEventuallyFormula().getSubformula())
                        ->asExplicitQualitativeCheckResult().getTruthValuesVector();
                absorbingStatesForSubFormula |= mopmc::graph::performProb1A(spModel, statesWithZeroReward, phi);

                std::cout << "Rewards model states flipped: " << statesWithZeroReward.getNumberOfSetBits() << "/ " << statesWithZeroReward.size() << std::endl;

            } else {
                throw std::runtime_error("Framework currently only considers eventually formulas");
            }


        } else {
            throw std::runtime_error("Could not process sub-formula: Multi-objective:reduceStates");
        }
        absorbingStates&=absorbingStatesForSubFormula;
        if (absorbingStates.empty()) {
            std::cout << "Absorbing states are empty\n";
            break;
        }
    }

    if (!absorbingStates.empty()) {
        // We can make the states absorbing and delete unreachable states.
        storm::storage::BitVector mopmcSubsystemActions(spModel.getNumberOfChoices(), true);

        for (auto absorbingState : absorbingStates) {
            std::vector<uint_fast64_t> const& actions = spModel.getNumberActionsForState(absorbingState);
            for( uint_fast64_t action : actions) {
                mopmcSubsystemActions.set(action, false);
            }
        }

        storm::storage::BitVector subSystemStates = mopmc::multiobj::getReachableSubSystem(
            spModel,
            storm::storage::BitVector(spModel.getNumberOfStates(), true),
            mopmcSubsystemActions
        );
        makeSubSystem(spModel, subSystemStates, mopmcSubsystemActions);
    }
}

template<typename T>
void mopmc::multiobj::makeSubSystem(
    sparse::SparseModelBuilder<T> &spModel,
    const storm::storage::BitVector &subsystemStates,
    const storm::storage::BitVector &subSystemActions) {

    // We can just say here that if the model has deadlock states
    // we do not deal with these in the current version and the model should
    // be specified correctly.

    // Initially we will just get some information about the deadlocks
    const typename sparse::SparseModelBuilder<T>::SpMat& transitionMatrix = spModel.getTransitionMatrix();
    storm::storage::BitVector keptActions(transitionMatrix.rows(), false);
    // can we make the assumption that all enabled actions lead only to the
    // subsystem? How do we answer this question?
    // if we loop over the enabled actions are there any of them which lead
    // to states not in the subsystem
    for (uint_fast64_t subsystemState: subsystemStates) {
        std::vector<uint_fast64_t>&actions = spModel.getActionsForState(subsystemState);
        bool hasDeadlock = true;
        bool allEntriesStayInSubsystem = true;
        for(uint_fast64_t action: actions) {
            if(subSystemActions.get(action)) {
                typename mopmc::sparse::SparseModelBuilder<T>::SpMat::InnerIterator it(transitionMatrix, action);
                for(; it; ++it) {
                    if (!subsystemStates[it.col()]) {
                        allEntriesStayInSubsystem = false;
                        std::cout << "State: " << subsystemState << " is a deadlock state\n";
                    }
                }
            }
            if (allEntriesStayInSubsystem) {
                keptActions.set(action, true);
                hasDeadlock = false;
            }
        }
        if (hasDeadlock) {
            throw std::runtime_error(
                "This version does not deal with deadlock states. "
                "Please specify a correct model");
        }
    }
}

template <typename T>
storm::storage::BitVector mopmc::multiobj::getReachableSubSystem(
        sparse::SparseModelBuilder<T> &spModel,
        const storm::storage::BitVector &subsystemStates,
        const storm::storage::BitVector &subsystemActions) {

    storm::storage::BitVector initialStates = spModel.getInitialStates() & subsystemStates;
    storm::storage::BitVector reachableStates = mopmc::graph::getReachableStates<T>(
        spModel,
        initialStates,
        subsystemStates,
        storm::storage::BitVector(subsystemStates.size(), false),
        subsystemActions);
    return reachableStates;
}

template <typename T>
storm::storage::BitVector mopmc::multiobj::getOnlyReachableViaPhi(mopmc::sparse::SparseModelBuilder<T>& spModel,
                                                 storm::storage::BitVector const& phi){
    boost::optional<storm::storage::BitVector> choiceFilter;
    storm::storage::BitVector result = mopmc::graph::getReachableStates<T>(
        spModel,
      spModel.getInitialStates(),
      ~phi,
      storm::storage::BitVector(phi.size(), false),
      choiceFilter);
    result.complement();
    std::cout << "Set bits in result: " << result.getNumberOfSetBits() << "\n";
    assert(phi.isSubsetOf(result));
    return result;
}

// Explicit Instantiation
template void mopmc::multiobj::performMultiObjectiveModelChecking<double>(
    mopmc::sparse::SparseModelBuilder<double> &spModel,
    const storm::logic::MultiObjectiveFormula &formula);

template void mopmc::multiobj::preprocess<double>(mopmc::sparse::SparseModelBuilder<double> &spModel,
    const storm::logic::MultiObjectiveFormula &formula);

template void mopmc::multiobj::reduceStates<double>(mopmc::sparse::SparseModelBuilder<double> &spModel,
    boost::optional<std::string> deadlockLabel,
    const storm::logic::MultiObjectiveFormula &formula
);

template storm::storage::BitVector mopmc::multiobj::getOnlyReachableViaPhi<double>(
    mopmc::sparse::SparseModelBuilder<double>& spModel,
    storm::storage::BitVector const& phi);

template storm::storage::BitVector mopmc::multiobj::getReachableSubSystem<double>(
        sparse::SparseModelBuilder <double> &spModel,
        const storm::storage::BitVector &subsystemStates,
        const storm::storage::BitVector &subsystemActions);

template void mopmc::multiobj::makeSubSystem(
    sparse::SparseModelBuilder<double> &spModel,
    const storm::storage::BitVector &subsystemStates,
    const storm::storage::BitVector &subSystemActions);
