//
// Created by thomas on 20/08/23.
//
#include "MultiObjectivePreprocessor.h"
#include <storm/storage/SchedulerClass.h>
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
#include <storm/utility/FilteredRewardModel.h>
#include <storm/logic/OperatorFormula.h>
#include <storm/utility/vector.h>
#include <storm/models/sparse/MarkovAutomaton.h>
#include "SparseMultiObjective.h"

namespace mopmc {
    namespace stormtest {
        template<class SparseModelType>
        typename SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType SparseMultiObjectivePreprocessor<SparseModelType>::preprocess(
            const storm::Environment &env,
            SparseModelType &originalModel,
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
                    throw std::runtime_error("The given scheduler restriction has not been implemented.");
                }
            } else {
                model = storm::transformer::MemoryIncorporation<SparseModelType>::incorporateGoalMemory(originalModel, originalFormula.getSubformulas());
            }

            boost::optional<std::string> deadlockLabel;
            removeIrrelevantStates(model, deadlockLabel, originalFormula);

            PreprocessorData data(model);

            // Invoke preprocessing on individual objectives
            for(auto const& subFormula : originalFormula.getSubformulas()) {
                std::cout << "Preprocessing objective: " << *subFormula << ".\n";
                data.objectives.push_back(std::make_shared<storm::modelchecker::multiobjective::Objective<ValueType>>());
                data.objectives.back()->originalFormula = subFormula;
                data.finiteRewardCheckObjectives.resize(data.objectives.size(), false);
                data.upperResultBoundObjectives.resize(data.objectives.size(), false);
                if(!data.objectives.back()->originalFormula->isOperatorFormula()) {
                    std::stringstream ss;
                    ss << "Could not preprocess the subformula "
                        << *subFormula << " of " << originalFormula << " because it is not supported";
                    throw std::runtime_error(ss.str());
                }
                preprocessOperatorFormula(data.objectives.back()->originalFormula->asOperatorFormula(), data);
            }

            // Remove reward models that are not needed anymore
            std::set<std::string> relevantRewardModels;
            for (auto const& obj : data.objectives) {
                obj->formula->gatherReferencedRewardModels(relevantRewardModels);
            }
            data.model->restrictRewardModels(relevantRewardModels);

            // Build the actual results
            return buildResult(originalModel, originalFormula, data);
        }

        template<class SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::removeIrrelevantStates(
            std::shared_ptr<SparseModelType> &model,
            boost::optional<std::string> &deadlockLabel,
            const storm::logic::MultiObjectiveFormula &originalFormula) {
            storm::storage::BitVector absorbingStates(model->getNumberOfStates(), true);

            storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> mc(*model);
            storm::storage::SparseMatrix<ValueType> backwardTransitions = model->getBackwardTransitions();


            for (auto const& opFormula : originalFormula.getSubformulas()) {
                storm::storage::BitVector absorbingStatesForSubFormula;
                if (!opFormula->isOperatorFormula()) {
                    std::cout << "Cannot process this formula\n";
                }

                auto const& pathFormula = opFormula->asOperatorFormula().getSubformula();
                std::cout << "Probability formula? " << (opFormula->isProbabilityOperatorFormula() ? "yes" : "no") <<std::endl;
                if(opFormula ->isProbabilityOperatorFormula()) {
                    if (pathFormula.isUntilFormula()) {
                        std::cout << "Path formula is until formula\n";
                        auto lhs = mc.check(pathFormula.asUntilFormula().getLeftSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
                        auto rhs = mc.check(pathFormula.asUntilFormula().getRightSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
                        absorbingStatesForSubFormula = storm::utility::graph::performProb0A(backwardTransitions, lhs, rhs);
                        absorbingStatesForSubFormula |= getOnlyReachableViaPhi(*model, ~lhs | rhs);
                    } else if (pathFormula.isBoundedUntilFormula()) {
                        std::cout<< "Bounded until formula\n";
                    } else if (pathFormula.isGloballyFormula()) {
                        std::cout << "Is global formula\n";

                        auto stormPhi = mc.check(pathFormula.asGloballyFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
                        auto notStormPhi = ~stormPhi;
                        absorbingStatesForSubFormula = storm::utility::graph::performProb0A(backwardTransitions, stormPhi, notStormPhi);
                        absorbingStatesForSubFormula |= getOnlyReachableViaPhi(*model, notStormPhi);
                        std::cout << "Finished backward check: globally\n";
                    } else if (pathFormula.isEventuallyFormula()) {
                        std::cout << "Is eventual formula\n";

                        auto stormPhi = mc.check(pathFormula.asEventuallyFormula().getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();
                        absorbingStatesForSubFormula = storm::utility::graph::performProb0A(backwardTransitions, ~stormPhi, stormPhi);
                        absorbingStatesForSubFormula |= getOnlyReachableViaPhi(*model, stormPhi);
                        std::cout << "Finished backward check: eventually\n";
                    } else {
                        throw std::runtime_error("Probability formula type not implemented");
                    }

                } else if (opFormula -> isRewardOperatorFormula()) {

                    auto const& baseRewardModel = opFormula->asRewardOperatorFormula().hasRewardModelName()
                            ? model->getRewardModel(opFormula->asRewardOperatorFormula().getRewardModelName())
                            : model->getUniqueRewardModel();

                    if (pathFormula.isEventuallyFormula()) {
                        std::cout << "Discrete Time Model? " << (model->isDiscreteTimeModel() ? "yes" : "no") << std::endl;
                        auto rewardModel = storm::utility::createFilteredRewardModel(
                                baseRewardModel, model->isDiscreteTimeModel(), pathFormula.asEventuallyFormula());

                        storm::storage::BitVector statesWithoutReward = rewardModel.get().getStatesWithZeroReward(
                                model->getTransitionMatrix());
                        // Make states that can not reach a state with non-zero reward absorbing
                        absorbingStatesForSubFormula = storm::utility::graph::performProb0A(
                                backwardTransitions, statesWithoutReward, ~statesWithoutReward);

                        auto phi = mc.check(pathFormula.asEventuallyFormula().getSubformula()) ->
                                asExplicitQualitativeCheckResult().getTruthValuesVector();
                        // Make states that reach phi with prob1 while only visiting states with reward 0 absorbing
                        absorbingStatesForSubFormula |= storm::utility::graph::performProb1A(
                                model->getTransitionMatrix(), model->getTransitionMatrix().getRowGroupIndices(),
                                backwardTransitions, statesWithoutReward, phi);
                        // Make states that are only reachable via phi absorbing
                        absorbingStatesForSubFormula |= getOnlyReachableViaPhi(*model, phi);
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
                storm::storage::BitVector subsystemActions(model->getNumberOfChoices(), true);
                for (auto absorbingState : absorbingStates) {
                    for (uint64_t action = model->getTransitionMatrix().getRowGroupIndices()[absorbingState];
                        action < model->getTransitionMatrix().getRowGroupIndices()[absorbingState + 1]; ++action) {
                        subsystemActions.set(action, false);
                    }
                }
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
            }
        }

        template<typename SparseModelType>
        storm::storage::BitVector SparseMultiObjectivePreprocessor<SparseModelType>::getOnlyReachableViaPhi(
                SparseModelType& model,
                storm::storage::BitVector const& phi) {
            // Get the complement of the states that are reachable without visiting phi
            auto result =
                    storm::utility::graph::getReachableStates(model.getTransitionMatrix(), model.getInitialStates(), ~phi, storm::storage::BitVector(phi.size(), false));
            result.complement();
            assert(phi.isSubsetOf(result));
            return result;
        }

        storm::logic::OperatorInformation getOperatorInformation(storm::logic::OperatorFormula const& formula,
                                                                 bool considerComplementaryEvent) {
            storm::logic::OperatorInformation opInfo;

            if(formula.hasBound()) {

                std::cout << "Formula has bound? " << (formula.hasBound() ? "yes" : "no") << "\n";
                opInfo.bound = formula.getBound();

                // Invert the bound (if necessary)
                if (considerComplementaryEvent) {
                    opInfo.bound->threshold = opInfo.bound->threshold.getManager().rational(1.0) - opInfo.bound->threshold;
                    switch (opInfo.bound->comparisonType) {
                        case storm::logic::ComparisonType::Greater:
                            opInfo.bound->comparisonType = storm::logic::ComparisonType::Less;
                            break;
                        case storm::logic::ComparisonType::GreaterEqual:
                            opInfo.bound->comparisonType = storm::logic::ComparisonType::LessEqual;
                            break;
                        case storm::logic::ComparisonType::Less:
                            opInfo.bound->comparisonType = storm::logic::ComparisonType::Greater;
                            break;
                        case storm::logic::ComparisonType::LessEqual:
                            opInfo.bound->comparisonType = storm::logic::ComparisonType::GreaterEqual;
                            break;
                        default:
                            std::stringstream ss;
                            ss << "Current objective " << formula << " ignored as the formula also specifies a threshold";
                            throw std::runtime_error(ss.str());
                    }
                }
                if (storm::logic::isLowerBound(opInfo.bound->comparisonType)) {
                    opInfo.optimalityType = storm::solver::OptimizationDirection::Maximize;
                } else {
                    opInfo.optimalityType = storm::solver::OptimizationDirection::Minimize;
                }
            } else if (formula.hasOptimalityType()) {
                opInfo.optimalityType = formula.getOptimalityType();
                // Invert the optimality type if necessary
                if (considerComplementaryEvent) {
                    opInfo.optimalityType = storm::solver::invert(opInfo.optimalityType.get());
                }
            } else {
                std::stringstream ss;
                ss << "Objective " << formula << " does not specify whether to minimise of maximise";
                throw std::runtime_error(ss.str());
            }
            return opInfo;
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessOperatorFormula(
            const storm::logic::OperatorFormula &formula,
            mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::PreprocessorData &data) {

            storm::modelchecker::multiobjective::Objective<ValueType>& objective = *data.objectives.back();

            // Check whether the complementary event is considered
            objective.considersComplementaryEvent = formula.isProbabilityOperatorFormula()
                && formula.getSubformula().isGloballyFormula();

            // Extract the operator information from the formula and potentially invert it for the complementary event
            storm::logic::OperatorInformation opInfo = getOperatorInformation(formula, objective.considersComplementaryEvent);

            if (formula.isProbabilityOperatorFormula()) {
                preprocessProbabilityOperatorFormula(formula.asProbabilityOperatorFormula(), opInfo, data);
            } else if (formula.isRewardOperatorFormula()) {
                preprocessRewardOperatorFormula(formula.asRewardOperatorFormula(), opInfo, data);
            } else {
                std::stringstream ss;
                ss << "Could not process the objective " << formula << " because it is not supported";
                throw std::runtime_error(ss.str());
            }
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessRewardOperatorFormula(storm::logic::RewardOperatorFormula const& formula,
                                                                                                storm::logic::OperatorInformation const& opInfo,
                                                                                                PreprocessorData& data) {
            std::string rewardModelName;
            if (formula.hasRewardModelName()) {
                rewardModelName = formula.getRewardModelName();

                if(!data.model->hasRewardModel(rewardModelName)){
                    std::stringstream ss;
                    ss << "The reward model specified by formula " << formula << " does not exist in the model";
                    throw std::runtime_error(ss.str());
                }
            } else {
                // We have to assert that a unique reward model exists, and we need to find its name.
                // However, we might have added auxiliary reward models for other objectives which we have to filter out here.
                auto prefixOf = [](std::string const& left, std::string const& right) {
                    return std::mismatch(left.begin(), left.end(), right.begin()).first == left.end();
                };
                bool uniqueRewardModelFound = false;
                for (auto const& rewModel : data.model->getRewardModels()) {
                    if (prefixOf(data.rewardModelNamePrefix, rewModel.first)) {
                        // Skip auxiliary reward model
                        continue;
                    }
                    if(uniqueRewardModelFound) {
                        std::stringstream ss;
                        ss << "The formula " << formula << " does not specify a reward model name and the reward model is not unique.";
                        throw std::runtime_error(ss.str());
                    }
                    uniqueRewardModelFound = true;
                    rewardModelName = rewModel.first;
                }

                if(!uniqueRewardModelFound) {
                    std::stringstream ss;
                    ss << "The formula " << formula << " refers to an unnamed reward model but no reward model has been defined.";
                    throw std::runtime_error(ss.str());
                }
            }

            data.objectives.back()->lowerResultBound = storm::utility::zero<ValueType>();

            if (formula.getSubformula().isEventuallyFormula()) {
                preprocessEventuallyFormula(formula.getSubformula().asEventuallyFormula(), opInfo, data, rewardModelName);
            } else if (formula.getSubformula().isCumulativeRewardFormula()) {
                preprocessCumulativeRewardFormula(formula.getSubformula().asCumulativeRewardFormula(), opInfo, data, rewardModelName);
            } else if (formula.getSubformula().isTotalRewardFormula()) {
                preprocessTotalRewardFormula(formula.getSubformula().asTotalRewardFormula(), opInfo, data, rewardModelName);
            } /*else if (formula.getSubformula().isLongRunAverageRewardFormula()) {
                preprocessLongRunAverageRewardFormula(formula.getSubformula().asLongRunAverageRewardFormula(), opInfo, data, rewardModelName);
            }*/
            else {
                std::stringstream ss;
                ss << "The subformula of " << formula << " is not supported.";
                throw std::runtime_error(ss.str());
            }
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessProbabilityOperatorFormula(storm::logic::ProbabilityOperatorFormula const& formula,
                                                                                                     storm::logic::OperatorInformation const& opInfo,
                                                                                                     PreprocessorData& data) {
            // Probabilities are between zero and one
            data.objectives.back()->lowerResultBound = storm::utility::zero<ValueType>();
            data.objectives.back()->upperResultBound = storm::utility::one<ValueType>();

            if (formula.getSubformula().isUntilFormula()) {
                preprocessUntilFormula(formula.getSubformula().asUntilFormula(), opInfo, data);
            } else if (formula.getSubformula().isBoundedUntilFormula()) {
                preprocessBoundedUntilFormula(formula.getSubformula().asBoundedUntilFormula(), opInfo, data);
            } else if (formula.getSubformula().isGloballyFormula()) {
                preprocessGloballyFormula(formula.getSubformula().asGloballyFormula(), opInfo, data);
            } else if (formula.getSubformula().isEventuallyFormula()) {
                preprocessEventuallyFormula(formula.getSubformula().asEventuallyFormula(), opInfo, data);
            } else {
                std::stringstream ss;
                ss << "The subformula of " << formula << " is not supported.";
                throw std::runtime_error(ss.str());
            }
        }

        template <typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessUntilFormula(
            const storm::logic::UntilFormula &formula, const storm::logic::OperatorInformation &opInfo,
            mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::PreprocessorData &data,
            std::shared_ptr<const storm::logic::Formula> subformula) {

            // Try to transform the formula to expected total (or cumulative) rewards

            storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> mc(*data.model);
            storm::storage::BitVector rightSubformulaResult = mc.check(formula.getRightSubformula())->
                    asExplicitQualitativeCheckResult().getTruthValuesVector();

            if((data.model->getInitialStates() & rightSubformulaResult).empty()) {
                std::stringstream ss;
                ss << "The probability for the objective " << *data.objectives.back()->originalFormula <<
                    " is always one as the rhs of the until formula is true in the initial state. Not implemented";
                throw std::runtime_error(ss.str());
            }

            // Whenever a state that violates the left subformula or satisfies the right subformula is reached,
            // the objective is 'decided', i.e. no more reward should be collected from there
            storm::storage::BitVector notLeftOrRight = mc.check(formula.getLeftSubformula())
                    ->asExplicitQualitativeCheckResult().getTruthValuesVector();
            notLeftOrRight.complement();
            notLeftOrRight |= rightSubformulaResult;

            // Get the states that are reachable from a notLeftOrRightState
            storm::storage::BitVector allStates(data.model->getNumberOfStates(), true), noStates(data.model->getNumberOfStates(), false);
            storm::storage::BitVector reachableFromGoal = storm::utility::graph::getReachableStates(
                    data.model->getTransitionMatrix(), notLeftOrRight, allStates, noStates);
            // Get the states that are reachable from an initial state, stopping at the states reachable from goal
            storm::storage::BitVector reachableFromInit =
                    storm::utility::graph::getReachableStates(data.model->getTransitionMatrix(), data.model->getInitialStates(),
                                                              ~notLeftOrRight, reachableFromGoal);
            // Exclude the actual notLeftOrRight states from the states that are reachable from init
            reachableFromInit &= ~notLeftOrRight;
            if ((reachableFromInit & reachableFromGoal).empty()) {
                std::cout << "Objective " << *data.objectives.back()->originalFormula <<
                    "is transformed to an expected total/.cumulative reward property.\n";
                // Transform to expected total rewards:
                // build a stateAction reward vector that gives (one*transitionProbability) reward whenever a transition
                // leads from a reachableFromInit state to a goal state
                std::vector<typename SparseModelType::ValueType> objectiveRewards(data.model->getTransitionMatrix().getRowCount(),
                                                                                  storm::utility::zero<typename SparseModelType::ValueType>());
                for (auto state: reachableFromInit) {
                    for(uint_fast64_t row = data.model->getTransitionMatrix().getRowGroupIndices()[state];
                        row < data.model->getTransitionMatrix().getRowGroupIndices()[state + 1]; ++row) {
                        objectiveRewards[row] = data.model->getTransitionMatrix().getConstrainedRowSum(row, rightSubformulaResult);
                    }
                }
                std::string rewardModelName = data.rewardModelNamePrefix + std::to_string(data.objectives.size());
                data.model->addRewardModel(rewardModelName, typename SparseModelType::RewardModelType(std::nullopt, std::move(objectiveRewards)));
                if(subformula == nullptr) {
                    subformula = std::make_shared<storm::logic::TotalRewardFormula>();
                }
                data.objectives.back()->formula = std::make_shared<storm::logic::RewardOperatorFormula>(subformula, rewardModelName, opInfo);
            } else {
                std::cout << "Objective " << *data.objectives.back()->originalFormula <<
                    " can not be transformed to an expected total/cumulative reward property.\n";
                data.objectives.back()->formula = std::make_shared<storm::logic::ProbabilityOperatorFormula>(formula.asSharedPointer(), opInfo);
            }
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessBoundedUntilFormula(
            const storm::logic::BoundedUntilFormula &formula, const storm::logic::OperatorInformation &opInfo,
            mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::PreprocessorData &data) {
            // Check how to handle this query
            if (formula.isMultiDimensional() || formula.getTimeBoundReference().isRewardBound()) {
                std::cout << "Formula is reward bounded or multidimensional. Objective "
                    << data.objectives.back()->originalFormula
                    << " is not transformed to an expected cumulative reward property.\n";
                data.objectives.back()->formula =
                        std::make_shared<storm::logic::ProbabilityOperatorFormula>(formula.asSharedPointer(), opInfo);
            } else  if (!formula.hasLowerBound() || (!formula.isLowerBoundStrict() && storm::utility::isZero(formula.template getLowerBound<typename SparseModelType::ValueType>()))){
                std::shared_ptr<storm::logic::Formula const> subformula;
                if(!formula.hasUpperBound()) {
                    subformula = std::make_shared<storm::logic::TotalRewardFormula>();
                } else {
                    if (!data.model->isOfType(storm::models::ModelType::MarkovAutomaton) || formula.getTimeBoundReference().isTimeBound()) {
                        throw std::runtime_error("Invalid property exception, bounded until formulas for markov automata are"
                                                 "only allowed when time bounds are considered");

                    }
                    storm::logic::TimeBound bound(formula.isUpperBoundStrict(), formula.getUpperBound());
                    subformula = std::make_shared<storm::logic::CumulativeRewardFormula>(bound, formula.getTimeBoundReference());
                }
                preprocessUntilFormula(storm::logic::UntilFormula(formula.getLeftSubformula().asSharedPointer(), formula.getRightSubformula().asSharedPointer()),
                                       opInfo, data, subformula);
            } else{
                std::stringstream ss;
                ss << "Property " << formula << " is not supported";
                throw std::runtime_error(ss.str());
            }
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessGloballyFormula(storm::logic::GloballyFormula const& formula,
                                                                                          storm::logic::OperatorInformation const& opInfo, PreprocessorData& data) {
            // The formula is transformed to an until formula for the complementary event.
            auto negatedSubformula = std::make_shared<storm::logic::UnaryBooleanStateFormula>(storm::logic::UnaryBooleanStateFormula::OperatorType::Not,
                                                                                              formula.getSubformula().asSharedPointer());

            preprocessUntilFormula(storm::logic::UntilFormula(storm::logic::Formula::getTrueFormula(), negatedSubformula), opInfo, data);
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessEventuallyFormula(
            const storm::logic::EventuallyFormula &formula, const storm::logic::OperatorInformation &opInfo,
            mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::PreprocessorData &data,
            const boost::optional<std::string> &optionalRewardModelName) {
            if (formula.isReachabilityProbabilityFormula()) {
                preprocessUntilFormula(
                    *std::make_shared<storm::logic::UntilFormula>(storm::logic::Formula::getTrueFormula(), formula.getSubformula().asSharedPointer()), opInfo, data);
                return;
            }

            // Analyze the subformula
            storm::modelchecker::SparsePropositionalModelChecker<SparseModelType> mc(*data.model);
            storm::storage::BitVector subFormulaResult = mc.check(formula.getSubformula())->asExplicitQualitativeCheckResult().getTruthValuesVector();

            // Get the states that are reachable from a goal state
            storm::storage::BitVector allStates(data.model->getNumberOfStates(), true), noStates(data.model->getNumberOfStates(), false);
            storm::storage::BitVector reachableFromGoal =
                    storm::utility::graph::getReachableStates(data.model->getTransitionMatrix(), subFormulaResult, allStates, noStates);
            // Get the states that are reachable from an initial state, stopping at the states reachable from goal
            storm::storage::BitVector reachableFromInit =
                    storm::utility::graph::getReachableStates(data.model->getTransitionMatrix(), data.model->getInitialStates(), allStates, reachableFromGoal);
            // Exclude the actual goal states from the states that are reachable from an initial state
            reachableFromInit &= ~subFormulaResult;
            // If we can reach a state that is reachable from goal but which is not a goal state, it means that the transformation to expected total rewards is not
            // possible.
            if ((reachableFromInit & reachableFromGoal).empty()) {
                std::cout << "Objective " << *data.objectives.back()->originalFormula << " is transformed to an expected total reward property.\n";
                // Transform to expected total rewards:

                std::string rewardModelName = data.rewardModelNamePrefix + std::to_string(data.objectives.size());
                auto totalRewardFormula = std::make_shared<storm::logic::TotalRewardFormula>();
                data.objectives.back()->formula = std::make_shared<storm::logic::RewardOperatorFormula>(totalRewardFormula, rewardModelName, opInfo);

                if (formula.isReachabilityRewardFormula()) {
                    // build stateAction reward vector that only gives reward for states that are reachable from init
                    assert(optionalRewardModelName.is_initialized());
                    auto objectiveRewards =
                            storm::utility::createFilteredRewardModel(data.model->getRewardModel(optionalRewardModelName.get()), data.model->isDiscreteTimeModel(), formula)
                                    .extract();
                    // get rid of potential transition rewards
                    objectiveRewards.reduceToStateBasedRewards(data.model->getTransitionMatrix(), false);
                    if (objectiveRewards.hasStateRewards()) {
                        storm::utility::vector::setVectorValues(objectiveRewards.getStateRewardVector(), reachableFromGoal,
                                                                storm::utility::zero<typename SparseModelType::ValueType>());
                    }
                    if (objectiveRewards.hasStateActionRewards()) {
                        for (auto state : reachableFromGoal) {
                            std::fill_n(objectiveRewards.getStateActionRewardVector().begin() + data.model->getTransitionMatrix().getRowGroupIndices()[state],
                                        data.model->getTransitionMatrix().getRowGroupSize(state), storm::utility::zero<typename SparseModelType::ValueType>());
                        }
                    }
                    data.model->addRewardModel(rewardModelName, std::move(objectiveRewards));
                } else if (formula.isReachabilityTimeFormula()) {
                    // build state reward vector that only gives reward for relevant states
                    std::vector<typename SparseModelType::ValueType> timeRewards(data.model->getNumberOfStates(),
                                                                                 storm::utility::zero<typename SparseModelType::ValueType>());
                    if (data.model->isOfType(storm::models::ModelType::MarkovAutomaton)) {
                        storm::utility::vector::setVectorValues(
                                timeRewards,
                                dynamic_cast<storm::models::sparse::MarkovAutomaton<typename SparseModelType::ValueType> const&>(*data.model).getMarkovianStates() &
                                reachableFromInit,
                                storm::utility::one<typename SparseModelType::ValueType>());
                    } else {
                        storm::utility::vector::setVectorValues(timeRewards, reachableFromInit, storm::utility::one<typename SparseModelType::ValueType>());
                    }
                    data.model->addRewardModel(rewardModelName, typename SparseModelType::RewardModelType(std::move(timeRewards)));
                } else {
                    std::stringstream ss;
                    ss << "The formula " << formula << " neither considers reachability probabilities nor reachability rewards "
                       << (data.model->isOfType(storm::models::ModelType::MarkovAutomaton) ? "nor reachability time" : "")
                       << ". This is not supported.";
                    throw std::runtime_error(ss.str());
                }
            } else {
                std::cout << "Objective " << *data.objectives.back()->originalFormula << " can not be transformed to an expected total/cumulative reward property.\n";
                if (formula.isReachabilityRewardFormula()) {
                    assert(optionalRewardModelName.is_initialized());
                    if (data.deadlockLabel) {
                        // We made some states absorbing and created a new deadlock state. To make sure that this deadlock state gets value zero, we add it to the set
                        // of goal states of the formula.
                        std::shared_ptr<storm::logic::Formula const> newSubSubformula =
                                std::make_shared<storm::logic::AtomicLabelFormula const>(data.deadlockLabel.get());
                        std::shared_ptr<storm::logic::Formula const> newSubformula = std::make_shared<storm::logic::BinaryBooleanStateFormula const>(
                                storm::logic::BinaryBooleanStateFormula::OperatorType::Or, formula.getSubformula().asSharedPointer(), newSubSubformula);
                        boost::optional<storm::logic::RewardAccumulation> newRewardAccumulation;
                        if (formula.hasRewardAccumulation()) {
                            newRewardAccumulation = formula.getRewardAccumulation();
                        }
                        std::shared_ptr<storm::logic::Formula const> newFormula =
                                std::make_shared<storm::logic::EventuallyFormula const>(newSubformula, formula.getContext(), newRewardAccumulation);
                        data.objectives.back()->formula = std::make_shared<storm::logic::RewardOperatorFormula>(newFormula, optionalRewardModelName.get(), opInfo);
                    } else {
                        data.objectives.back()->formula =
                                std::make_shared<storm::logic::RewardOperatorFormula>(formula.asSharedPointer(), optionalRewardModelName.get(), opInfo);
                    }
                } else if (formula.isReachabilityTimeFormula()) {
                    // Reduce to reachability rewards so that time formulas do not have to be treated seperately later.
                    std::string rewardModelName = data.rewardModelNamePrefix + std::to_string(data.objectives.size());
                    std::shared_ptr<storm::logic::Formula const> newSubformula = formula.getSubformula().asSharedPointer();
                    if (data.deadlockLabel) {
                        // We made some states absorbing and created a new deadlock state. To make sure that this deadlock state gets value zero, we add it to the set
                        // of goal states of the formula.
                        std::shared_ptr<storm::logic::Formula const> newSubSubformula =
                                std::make_shared<storm::logic::AtomicLabelFormula const>(data.deadlockLabel.get());
                        newSubformula = std::make_shared<storm::logic::BinaryBooleanStateFormula const>(storm::logic::BinaryBooleanStateFormula::OperatorType::Or,
                                                                                                        formula.getSubformula().asSharedPointer(), newSubSubformula);
                    }
                    auto newFormula = std::make_shared<storm::logic::EventuallyFormula>(newSubformula, storm::logic::FormulaContext::Reward);
                    data.objectives.back()->formula = std::make_shared<storm::logic::RewardOperatorFormula>(newFormula, rewardModelName, opInfo);
                    std::vector<typename SparseModelType::ValueType> timeRewards;
                    if (data.model->isOfType(storm::models::ModelType::MarkovAutomaton)) {
                        timeRewards.assign(data.model->getNumberOfStates(), storm::utility::zero<typename SparseModelType::ValueType>());
                        storm::utility::vector::setVectorValues(
                                timeRewards,
                                dynamic_cast<storm::models::sparse::MarkovAutomaton<typename SparseModelType::ValueType> const&>(*data.model).getMarkovianStates(),
                                storm::utility::one<typename SparseModelType::ValueType>());
                    } else {
                        timeRewards.assign(data.model->getNumberOfStates(), storm::utility::one<typename SparseModelType::ValueType>());
                    }
                    data.model->addRewardModel(rewardModelName, typename SparseModelType::RewardModelType(std::move(timeRewards)));
                } else {

                    std::stringstream ss;
                    ss << "The formula " << formula << " neither considers reachability probabilities nor reachability rewards "
                       << (data.model->isOfType(storm::models::ModelType::MarkovAutomaton) ? "nor reachability time" : "")
                       << ". This is not supported.";
                    throw std::runtime_error(ss.str());
                }
            }
            data.finiteRewardCheckObjectives.set(data.objectives.size() - 1, true);
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessCumulativeRewardFormula(storm::logic::CumulativeRewardFormula const& formula,
                                                                                                  storm::logic::OperatorInformation const& opInfo,
                                                                                                  PreprocessorData& data,
                                                                                                  boost::optional<std::string> const& optionalRewardModelName) {
            if (!data.model->isOfType(storm::models::ModelType::Mdp)){
                throw std::runtime_error("Cumulative reward formulas are not supported for the given model type.");
            }
            std::string rewardModelName = optionalRewardModelName.get();
            // Strip away potential RewardAccumulations in the formula itself but also in reward bounds
            auto filteredRewards = storm::utility::createFilteredRewardModel(data.model->getRewardModel(rewardModelName), data.model->isDiscreteTimeModel(), formula);
            if (filteredRewards.isDifferentFromUnfilteredModel()) {
                std::string rewardModelName = data.rewardModelNamePrefix + std::to_string(data.objectives.size());
                data.model->addRewardModel(rewardModelName, std::move(filteredRewards.extract()));
            }

            std::vector<storm::logic::TimeBoundReference> newTimeBoundReferences;
            bool onlyRewardBounds = true;
            for (uint64_t i = 0; i < formula.getDimension(); ++i) {
                auto oldTbr = formula.getTimeBoundReference(i);
                if (oldTbr.isRewardBound()) {
                    if (oldTbr.hasRewardAccumulation()) {
                        auto filteredBoundRewards = storm::utility::createFilteredRewardModel(data.model->getRewardModel(oldTbr.getRewardName()),
                                                                                              oldTbr.getRewardAccumulation(), data.model->isDiscreteTimeModel());
                        if (filteredBoundRewards.isDifferentFromUnfilteredModel()) {
                            std::string freshRewardModelName =
                                    data.rewardModelNamePrefix + std::to_string(data.objectives.size()) + std::string("_" + std::to_string(i));
                            data.model->addRewardModel(freshRewardModelName, std::move(filteredBoundRewards.extract()));
                            newTimeBoundReferences.emplace_back(freshRewardModelName);
                        } else {
                            // Strip away the reward accumulation
                            newTimeBoundReferences.emplace_back(oldTbr.getRewardName());
                        }
                    } else {
                        newTimeBoundReferences.push_back(oldTbr);
                    }
                } else {
                    onlyRewardBounds = false;
                    newTimeBoundReferences.push_back(oldTbr);
                }
            }

            auto newFormula = std::make_shared<storm::logic::CumulativeRewardFormula>(formula.getBounds(), newTimeBoundReferences);
            data.objectives.back()->formula = std::make_shared<storm::logic::RewardOperatorFormula>(newFormula, rewardModelName, opInfo);

            if (onlyRewardBounds) {
                data.finiteRewardCheckObjectives.set(data.objectives.size() - 1, true);
            }
        }

        template<typename SparseModelType>
        void SparseMultiObjectivePreprocessor<SparseModelType>::preprocessTotalRewardFormula(storm::logic::TotalRewardFormula const& formula,
                                                                                             storm::logic::OperatorInformation const& opInfo, PreprocessorData& data,
                                                                                             boost::optional<std::string> const& optionalRewardModelName) {
            std::string rewardModelName = optionalRewardModelName.get();
            auto filteredRewards = storm::utility::createFilteredRewardModel(data.model->getRewardModel(rewardModelName), data.model->isDiscreteTimeModel(), formula);
            if (filteredRewards.isDifferentFromUnfilteredModel()) {
                std::string rewardModelName = data.rewardModelNamePrefix + std::to_string(data.objectives.size());
                data.model->addRewardModel(rewardModelName, filteredRewards.extract());
            }
            data.objectives.back()->formula = std::make_shared<storm::logic::RewardOperatorFormula>(formula.stripRewardAccumulation(), rewardModelName, opInfo);
            data.finiteRewardCheckObjectives.set(data.objectives.size() - 1, true);
        }

        template<typename SparseModelType>
        typename SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType::QueryType SparseMultiObjectivePreprocessor<SparseModelType>::getQueryType(
            const std::vector<storm::modelchecker::multiobjective::Objective<ValueType>> &objectives) {
            uint_fast64_t numOfObjectivesWithThreshold = 0;
            for (auto& obj : objectives) {
                if (obj.formula->hasBound()) {
                    ++numOfObjectivesWithThreshold;
                }
            }
            if (numOfObjectivesWithThreshold == objectives.size()) {
                return ReturnType::QueryType::Achievability;
            } else if (numOfObjectivesWithThreshold + 1 == objectives.size()) {
                // Note: We do not want to consider a Pareto query when the total number of objectives is one.
                return ReturnType::QueryType::Quantitative;
            } else if (numOfObjectivesWithThreshold == 0) {
                return ReturnType::QueryType::Pareto;
            } else {
                std::stringstream ss;
                ss << "Invalid Multi-objective query: The number of qualitative objectives should be either 0 (Pareto query), 1 (quantitative query), or "
                      "#objectives (achievability query).";
                throw std::runtime_error(ss.str());
            }
        }

        template<typename SparseModelType>
        typename SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType SparseMultiObjectivePreprocessor<SparseModelType>::buildResult(
                SparseModelType& originalModel, storm::logic::MultiObjectiveFormula const& originalFormula, PreprocessorData& data) {
            ReturnType result(originalFormula, originalModel);
            auto backwardTransitions = data.model->getBackwardTransitions();
            result.preprocessedModel = data.model;

            for (auto& obj : data.objectives) {
                result.objectives.push_back(std::move(*obj));
            }
            result.queryType = getQueryType(result.objectives);
            result.maybeInfiniteRewardObjectives = std::move(data.finiteRewardCheckObjectives);

            return result;
        }

        template class SparseMultiObjectivePreprocessor<storm::models::sparse::Mdp<double>>;

        template<class SparseModelType>
        SparseMultiObjectivePreprocessor<SparseModelType>::PreprocessorData::PreprocessorData(
                std::shared_ptr<SparseModelType> model) : model(model) {
            // The rewardModelNamePrefix should be a prefix of a reward model of the given
            // model to ensure uniqueness of new reward model names
            rewardModelNamePrefix = "obj";
            while (true){
                bool prefixIsUnique = true;
                for (auto const& rewardModels : model->getRewardModels()) {
                    if (rewardModelNamePrefix.size() <= rewardModels.first.size()) {
                        if (std::mismatch(rewardModelNamePrefix.begin(), rewardModelNamePrefix.end(),
                                          rewardModels.first.begin()).first == rewardModelNamePrefix.end()) {
                            prefixIsUnique = false;
                            rewardModelNamePrefix = "_" + rewardModelNamePrefix;
                            break;
                        }
                    }
                }
                if (prefixIsUnique) {
                    break;
                }
            }

        }
    }
}