//
// Created by thomas on 20/08/23.
//
#include <storm/storage/SchedulerClass.h>
#include "mopmc-src/model-checking/MultiObjectivePreprocessor.h"
#include "ModelChecker.h"
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/transformer/MemoryIncorporation.h>
#include <storm/exceptions/NotImplementedException.h>
#include <storm/environment/modelchecker/ModelCheckerEnvironment.h>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/modelchecker/propositional/SparsePropositionalModelChecker.h>
#include <storm/modelchecker/results/ExplicitQualitativeCheckResult.h>
#include <storm/modelchecker/results/ExplicitQuantitativeCheckResult.h>
#include <storm/transformer/SubsystemBuilder.h>
#include <storm/utility/graph.h>

#include "mopmc-src/model-checking/GraphAnalysis.h"
#include "MultiObjective.h"
#include "MultiObjectiveTest.h"

namespace mopmc {
    namespace stormtest {
        template<class SparseModelType>
        //typename SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocess(
            const storm::Environment &env,
            const SparseModelType &originalModel,
            const storm::logic::MultiObjectiveFormula &originalFormula
            //mopmc::sparse::SparseModelBuilder<double>& spModel
            ) {

            std::shared_ptr<SparseModelType> model;

            // Incorporate the necessary memory
            if (env.modelchecker().multi().isSchedulerRestrictionSet()) {
                auto const& schedRestr = env.modelchecker().multi().getSchedulerRestriction();
                if (schedRestr.getMemoryPattern() == storm::storage::SchedulerClass::MemoryPattern::GoalMemory) {
                    model = storm::transformer::MemoryIncorporation<SparseModelType>::incorporateGoalMemory(originalModel, originalFormula.getSubformulas());
                } else if (schedRestr.getMemoryPattern() == storm::storage::SchedulerClass::MemoryPattern::Arbitrary && schedRestr.getMemoryStates() > 1) {
                    model = storm::transformer::MemoryIncorporation<SparseModelType>::incorporateFullMemory(originalModel, schedRestr.getMemoryStates());
                } else if (schedRestr.getMemoryPattern() == storm::storage::SchedulerClass::MemoryPattern::Counter && schedRestr.getMemoryStates() > 1) {
                    model = storm::transformer::MemoryIncorporation<SparseModelType>::incorporateCountingMemory(originalModel, schedRestr.getMemoryStates());
                } else if (schedRestr.isPositional()) {
                    model = std::make_shared<SparseModelType>(originalModel);
                } else {
                    STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "The given scheduler restriction has not been implemented.");
                }
            } else {
                model = storm::transformer::MemoryIncorporation<SparseModelType>::incorporateGoalMemory(originalModel, originalFormula.getSubformulas());
            }

            boost::optional<std::string> deadlockLabel;
            removeIrrelevantStates(model, deadlockLabel, originalFormula);
        }

        template<class SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::removeIrrelevantStates(
            std::shared_ptr<SparseModelType> &model,
            boost::optional<std::string> &deadlockLabel,
            const storm::logic::MultiObjectiveFormula &originalFormula
            //mopmc::sparse::SparseModelBuilder<double>& spModel
            ) {
            storm::storage::BitVector absorbingStates(spModel.getNumberOfStates(), true);
            storm::storage::BitVector stormAbsorbingStates(model->getNumberOfStates(), true);

            storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> mc(*model);
            storm::storage::SparseMatrix<ValueType> backwardTransitions = model->getBackwardTransitions();

            auto mopmcModel = mopmc::model_checking::DTMCModelSolver<double>(spModel);

            for (auto const& opFormula : originalFormula.getSubformulas()) {
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

                        auto stormPhi = mc.check(pathFormula.asGloballyFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
                        auto notStormPhi = ~stormPhi;
                        stormAbsorbingStatesForSubFormula = storm::utility::graph::performProb0A(backwardTransitions, stormPhi, notStormPhi);
                        stormAbsorbingStatesForSubFormula |= getOnlyReachableViaPhi(*model, notStormPhi);

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

                        auto stormPhi = mc.check(pathFormula.asEventuallyFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
                        stormAbsorbingStatesForSubFormula = storm::utility::graph::performProb0A(backwardTransitions, ~stormPhi, stormPhi);
                        stormAbsorbingStatesForSubFormula |= getOnlyReachableViaPhi(*model, stormPhi);

                        if (stormAbsorbingStatesForSubFormula != absorbingStatesForSubFormula) {
                            std::cout << "The Globally subformula check produced the same bitVector\n";
                        } else {
                            std::cout << "The Globally subformula check did not produce the same bitVector\n";
                        }

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

                } else {
                    throw std::runtime_error("Could not process sub-formula: Multi-objective:reduceStates");
                }
                absorbingStates&=absorbingStatesForSubFormula;
                stormAbsorbingStates&=stormAbsorbingStatesForSubFormula;
                std::cout << "Storm absorbing equals MOPMC absorbing? " <<
                    (stormAbsorbingStates == absorbingStates ? "yes" : "no") << std::endl;

                if (absorbingStates.empty()) {
                    std::cout << "Absorbing states are empty\n";
                    break;
                }
            }

            if (!absorbingStates.empty()) {
                // We can make the states absorbing and delete unreachable states.
                storm::storage::BitVector subsystemActions(model->getNumberOfChoices(), true);
                storm::storage::BitVector mopmcSubsystemActions(spModel.getNumberOfChoices(), true);
                std::cout << "Subsystem actions size: " << mopmcSubsystemActions.size() << "\n";
                std::cout << "Subsystem actions equal? "
                    << (subsystemActions == mopmcSubsystemActions ? "yes" : "no") << std::endl;
                for (auto absorbingState : stormAbsorbingStates) {
                    std::cout << "absorbing state: " << absorbingState << " actions deleted: ";
                    for (uint64_t action = model->getTransitionMatrix().getRowGroupIndices()[absorbingState];
                         action < model->getTransitionMatrix().getRowGroupIndices()[absorbingState + 1]; ++action) {
                        //std::cout << "delete state: " << absorbingState << ", action: " << action << "\n";
                        subsystemActions.set(action, false);
                        std::cout << action << ", ";
                    }
                    std::cout << "\n";
                }
                std::cout << "MOPMC\n";
                for (auto absorbingState : absorbingStates) {
                    std::vector<uint_fast64_t> const& actions = spModel.getNumberActionsForState(absorbingState);
                    std::cout << "absorbing state: " << absorbingState << " actions deleted: ";
                    for( uint_fast64_t action : actions) {
                        mopmcSubsystemActions.set(action, false);
                        std::cout << action << ", ";
                    }
                    std::cout << "\n";
                }
                std::cout << "MOMPC number set bits: " << mopmcSubsystemActions.getNumberOfSetBits() << std::endl;
                std::cout << "Sub system actions: " << (mopmcSubsystemActions == subsystemActions ? "yes" : "no") << std::endl;
                storm::transformer::SubsystemBuilderOptions options;

                std::cout << "Initial states: " << model->getInitialStates() << std::endl;
                storm::storage::BitVector initialStates = model->getInitialStates() & storm::storage::BitVector(model->getNumberOfStates());
                std::cout << "Initial states set bits: " << initialStates.getNumberOfSetBits() << "\n";
                options.fixDeadlocks = true;
                auto const& submodel = storm::transformer::buildSubsystem(
                        *model, storm::storage::BitVector(model->getNumberOfStates(), true), subsystemActions, false, options);

                std::cout << "Making states absorbing reduced the state space from " << model->getNumberOfStates() << " to " << submodel.model->getNumberOfStates()
                                                                                       << "\n";
                model = submodel.model->template as<SparseModelType>();
                deadlockLabel = submodel.deadlockLabel;

                storm::storage::BitVector mopmcSubsystemStates = mopmc::multiobj::getReachableSubSystem(
                    spModel,
                    storm::storage::BitVector(spModel.getNumberOfStates(), true),
                    mopmcSubsystemActions
                );

                spModel.getMDPSubMatrix(mopmcSubsystemStates, mopmcSubsystemActions);
            }
        }

        template<typename SparseModelType>
        storm::storage::BitVector SparseMultiObjectivePreprocessor<SparseModelType>::getOnlyReachableViaPhi(
                SparseModelType const& model,
                storm::storage::BitVector const& phi) {
            // Get the complement of the states that are reachable without visiting phi
            auto result =
                    storm::utility::graph::getReachableStates(model.getTransitionMatrix(), model.getInitialStates(), ~phi, storm::storage::BitVector(phi.size(), false));
            result.complement();
            assert(phi.isSubsetOf(result));
            return result;
        }

        template class SparseMultiObjectivePreprocessor<storm::models::sparse::Mdp<double>>;
    }
}