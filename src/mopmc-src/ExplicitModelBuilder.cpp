//
// Created by thomas on 1/08/23.
//
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>

#include <storm/utility/initialize.h>

//#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/builder/StateAndChoiceInformationBuilder.h>
#include <storm/storage/sparse/StateStorage.h>

#include <storm/settings/modules/BuildSettings.h>

#include <storm/exceptions/AbortException.h>
#include <storm/utility/SignalHandler.h>

#include "ExplicitModelBuilder.h"
//#include "TransitionMatrixBuilder.h"
//#include "RewardModelBuilder.h"
#include "SparseModel.h"
#include "model-checking/ModelChecker.h"
#include <storm/utility/constants.h>
#include <Eigen/Sparse>
// std
#include <string>
#include <stdexcept>
#include <iostream>

/*
 * Calls storm::model_checker::AbstractModelChecker<ModelType = DTMC>::check(
 *  Environment const& env, CheckTask<storm::logic::Formula, ValueType> const& checkTask)
 * Within the above, storm checks what type of formula is input by the user
 *   storm::logic::Formula const& formula = checkTask.getFormula();
 * If the formula is a state formula then we proceed, otherwise we throw an error
 *
 *  At this point we move to AbstractModelChecker::checkStateFormula, where storm then
 *  determines what type of formula it is -> as we will only be dealing with
 *  LTL probability formula, the mompmc code will just do stateFormula.isBooleanLiteralFormula()
 *  check an then progresses to this->checkProbabilityOperatorFormula(
 *      env, checkTask.substituteFormula(stateFormula.asProbabilityOperatorFormula())
 *
 * In the above storm computes this->computeProbabilities(
 *  env, checkTask.substituteFormula(stateFormula.getSubformula()))
 *
 * computeProbabilities() is just a strategy function which calls the right modelBuilder checking
 * routine based on the type of formula. There are some checks to do here:
 *  1.
 */

bool mopmc::check(std::string const& path_to_model, std::string const& property_string) {
    
    // Assumes that the modelBuilder is in the prism program language format and parses the program.
    auto program = storm::parser::PrismParser::parse(path_to_model);
    // Code snippet assumes a Dtmc
    //assert(program.getModelType() == storm::prism::Program::ModelType::DTMC);
    std::cout << "Model Type: " << program.getModelType() << std::endl;
    // Then parse the properties, passing the program to give context to some potential variables.
    auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
    // Translate properties into the more low-level formulae.
    auto formulae = storm::api::extractFormulasFromProperties(properties);

    // Now translate the prism program into a DTMC in the sparse format.
    // Use the formulae to add the correct labelling.
    // This is the original storm helper code.
    //auto modelBuilder = storm::api::buildSparseModel<double>(program, formulae)->template as<Dtmc>();

    // My experiments
    
    storm::builder::BuilderOptions options(formulae, program);
    options.setBuildAllLabels();
    options.setBuildAllRewardModels();
    std::shared_ptr<storm::generator::NextStateGenerator<double>> generator;
    generator = std::make_shared<storm::generator::PrismNextStateGenerator<double>>(program, options);
    mopmc::ExplicitModelBuilder<double> modelBuilder(generator);
    bool deterministicModel = generator -> isDeterministicModel();
    //std::vector<mopmc::RewardModelBuilder<double>> rewardModelBuilders;
    //std::cout << "Number of reward models: " << generator -> getNumberOfRewardModels() << "\n";

    /*for (uint32_t i = 0; i < generator -> getNumberOfRewardModels(); i++) {
        rewardModelBuilders.emplace_back(generator->getRewardModelInformation(i));
    }*/

    //std::cout << "Reward modelBuilder 0 has state rewards: " << rewardModelBuilders[0].hasStateRewards();
    //SparseMatrixBuilder transitionMatrixBuilder(0, 0, 0, false, !deterministicModel, 0);

    // Declare a sparse matrix
    mopmc::sparse::SparseModelBuilder<double> spModel;

    storm::builder::StateAndChoiceInformationBuilder stateAndChoiceInformationBuilder;
    stateAndChoiceInformationBuilder.setBuildChoiceLabels(generator->getOptions().isBuildChoiceLabelsSet());
    stateAndChoiceInformationBuilder.setBuildChoiceOrigins(generator->getOptions().isBuildChoiceOriginsSet());
    
    std::cout << "Build state valuation set: "
    << static_cast<bool>(generator -> getOptions().isBuildStateValuationsSet())
    << std::endl;

    stateAndChoiceInformationBuilder.setBuildStateValuations(generator->getOptions().isBuildStateValuationsSet());


    modelBuilder.buildMatrices(spModel, stateAndChoiceInformationBuilder);

    // Build the state labelling after the all of the states have been explored and the hashmap of
    // seached states has been filled in.
    spModel.setStateLabels(modelBuilder.buildStateLabelling());

    auto& transitionMatrix = spModel.getSparseMatrix();
    std::cout << transitionMatrix.toDense() << std::endl;
    std::cout << "Matrix transpose: \n";
    std::cout << transitionMatrix.transpose().toDense() << std::endl;
    // Ok now that we have these three we can start investigating the build matrices routine
    //Create a callback for the nest-state generator to enable it to request the index of the states

    std::cout << "Has choice labels: " <<
      (stateAndChoiceInformationBuilder.isBuildChoiceLabels() ? "yes" : "no") << std::endl;
    
    // Create a modelBuilder checker on top of the sparse engine.
    auto abstractFormula = storm::modelchecker::CheckTask<>(*(formulae[0]), true);
    storm::logic::Formula const& form = abstractFormula.getFormula();
    std::cout << "state formula: " << (form.isStateFormula() ? "yes" : "no") << "\n";


    storm::modelchecker::CheckTask<storm::logic::StateFormula, double> const& checkTask =
            abstractFormula.substituteFormula(form.asStateFormula());

    // AbstractModelChecker::checkStateFormula()-v
    storm::logic::StateFormula const& stateFormula = checkTask.getFormula();

    std::cout << "Formula type: " << 
        (stateFormula.isProbabilityOperatorFormula() ? "yes" : "no") << "\n";

    std::cout << "Multi-objective: " << (stateFormula.isMultiObjectiveFormula() ? "yes" : "no") << std::endl;

    /*
    // At this point I think it is clear that mopmc needs to create its own AbstractModelChecker
    // class which inherits from storm AbstractModelChecker

    // make sure the spModel has a getNumberOfStatesMethod
    mopmc::model_checking::DTMCModelSolver<double> dtmcSolver(spModel);

    std::unique_ptr<storm::modelchecker::CheckResult> result =
            dtmcSolver.check(storm::modelchecker::CheckTask<>(*(formulae[0]), true));

    //auto checker = std::make_shared<DtmcModelChecker>(*modelBuilder);
    // Create a check task with the formula. Run this task with the modelBuilder checker.
    //auto result = checker->check(storm::modelchecker::CheckTask<>(*(formulae[0]), true));
    //assert(result->isExplicitQuantitativeCheckResult());
    // Use that we know that the modelBuilder checker produces an explicit quantitative result
    //auto quantRes = result->asExplicitQuantitativeCheckResult<double>();
    // TODO handle both qualitative and quantitative results
    std::cout << "Is explicit quant result? " << result->isExplicitQuantitativeCheckResult() << std::endl;
    std::cout << "Is explicit qual result? " << result->isExplicitQualitativeCheckResult() << std::endl;
    auto qualresult = result->asExplicitQualitativeCheckResult();
    std::cout << "Result: " << (qualresult[*spModel.getInitialStates().begin()] ? "true" : "false") << std::endl;
    // Now compare the result at the first initial state of the modelBuilder with 0.5.
    //return quantRes[*spModel.getInitialStates().begin()] > 0.5;
    */
    return true;
}

template<typename ValueType, typename StateType>
void mopmc::ExplicitModelBuilder<ValueType, StateType>::buildMatrices(
    mopmc::sparse::SparseModelBuilder<ValueType>& spModelBuilder,
    storm::builder::StateAndChoiceInformationBuilder& stateAndChoiceInformationBuilder
) {
    // Create a callback function for the next state generator to enable it to request the index of states
    std::function<StateType(CompressedState const&)> stateToIdCallback = 
        std::bind(&ExplicitModelBuilder<ValueType, StateType>::getOrAddStateIndex, this, std::placeholders::_1);
    
    // Assume that the exploration order is BFS
    /*for(auto& rewardModelBuilder : rewardModelBuilders) {
        std::cout << "Has state rewards: " << rewardModelBuilder.hasStateRewards() << "\n";
        std::cout << "Has state action rewards: " << rewardModelBuilder.hasStateActionRewards() << "\n";
    }*/
    this->stateStorage.initialStateIndices = generator->getInitialStates(stateToIdCallback);
    std::cout << "Initial states: ";
    for(uint32_t state : stateStorage.initialStateIndices) {
        std::cout << state << std::endl;
    }
    std::cout << "\n";

    auto timeOfStart = std::chrono::high_resolution_clock::now();
    // Explore the current state until there are no more reachable states
    uint_fast64_t currentRowGroup = 0;
    uint_fast64_t currentRow = 0;
    uint_fast64_t numberOfTransitions = 0;

    uint64_t numberOfExploredStates = 0;
    uint64_t numberOfExploredStatesSinceLastMessage = 0;
    bool printQueue = false;

    // Triplet list for the sparse matrix in row-major format
    std::vector<Eigen::Triplet<ValueType>> tripletList;

    while (!statesToExplore.empty()) {
        if (printQueue) {
            for(int i(statesToExplore.size()-1); i >= 0; i--) {
                std::cout << statesToExplore[i].second << " "; 
            }
        }
        // get the first state in the queue
        mopmc::CompressedState currentState = statesToExplore.front().first;
        StateType currentIndex = statesToExplore.front().second;
        statesToExplore.pop_front();

        generator -> load(currentState);
        // this call adds states to the deque
        storm::generator::StateBehavior<double, uint32_t> behaviour = generator->expand(stateToIdCallback);

        if (behaviour.empty()) {
            // End state behaviour
            if (!storm::settings::getModule<storm::settings::modules::BuildSettings>().isDontFixDeadlocksSet() || !behaviour.wasExpanded()) {
                if (behaviour.wasExpanded()) {
                    // pay attention to this -> we are storing the states somewhere that we have looked at already
                    // this is a part of the explicit model builder and we want to gain access to these private
                    // variables to better understand what is happening in the BFS
                    this -> stateStorage.deadlockStateIndices.push_back(currentIndex);
                }

                if (!generator -> isDeterministicModel()) {
                    // add in a new group for non-deterministic models into the sparse matrix
                }

                tripletList.emplace_back(currentRow, currentIndex, static_cast<ValueType>(1.0));

                /*
                for(auto& rewardModelBuilder : rewardModelBuilders) {
                    if (rewardModelBuilder.hasStateRewards()) {
                        rewardModelBuilder.addStateReward(storm::utility::zero<ValueType>());
                    }
                }                
                */

                std::cout << "state idx: " << currentIndex << ", current matrix row: "  << currentRow <<
                ", Current row group: " << currentRowGroup << ", Self Loop: Pairs: " << "(" << currentIndex << ", 1)\n";
                ++currentRow;
                ++currentRowGroup;
                ++numberOfTransitions;

            }
        } else {
            // Add the state rewards to the corresponding reward model

            auto stateRewardIt = behaviour.getStateRewards().begin();
            /*for(auto& rewardModelBuilder : rewardModelBuilders) {
                if(rewardModelBuilder.hasStateRewards()) {
                    rewardModelBuilder.addStateReward(*stateRewardIt);
                }
                ++stateRewardIt;
            }*/

            bool firstChoiceOfStates = true;
            int choices = static_cast<int>(behaviour.getNumberOfChoices());
            std::cout << "Number of choices: " << choices << std::endl;
            for (auto const& choice : behaviour) {
                // add the generated choic information
                std::cout << "Choice Labels: " << stateAndChoiceInformationBuilder.isBuildChoiceLabels() << std::endl;
                std::cout << "Has labels: " << choice.hasLabels() <<  std::endl;

                std::cout << "Choice Origins: " << stateAndChoiceInformationBuilder.isBuildChoiceOrigins() << std::endl;
                std::cout << "Origin Data: " << choice.hasOriginData() << std::endl;

                std::cout << "state idx: " << currentIndex << ", current matrix row: "  << currentRow
                << ", Current row group: " <<  currentRowGroup << ", Pairs: ";

                for (auto const& stateProbabilityPair : choice) {
                    std::cout << "(" << stateProbabilityPair.first << "," << stateProbabilityPair.second << "), ";
                    tripletList.emplace_back(currentRow, stateProbabilityPair.first, stateProbabilityPair.second);
                    ++numberOfTransitions;
                }
                std::cout << "\n";

                // Add the state-action rewards to rewards model
                std::cout << "Reward: ";
                auto choiceRewardIt = choice.getRewards();
                for (double val: choiceRewardIt) {
                    std::cout << val;
                }
                std::cout << "\n";
                /*
                for(auto& rewardModelBuilder : rewardModelBuilders) {
                    if(rewardModelBuilder.hasStateActionRewards()) {
                        rewardModelBuilder.addStateActionReward(*choiceRewardIt);
                    }
                    ++choiceRewardIt;
                }
                */
                ++currentRow;
                firstChoiceOfStates = false;
            }
            ++currentRowGroup;
        }
        ++numberOfExploredStates;

        if (storm::utility::resources::isTerminate()) {
            auto durationSinceStart = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - timeOfStart).count();
            std::cout << "Explored " << numberOfExploredStates << " states in " << durationSinceStart << " seconds before abort.\n";
            STORM_LOG_THROW(false, storm::exceptions::AbortException, "Aborted in state space exploration.");
            break;
        }
    }

    for (auto t : tripletList) {
        std::cout << "(" << t.row() << "," << t.col() << "," << t.value() << "), ";
    }
    spModelBuilder.setNumberOfStates(numberOfExploredStates);
    spModelBuilder.setNumberOfTransitions(numberOfTransitions);
    std::cout << "\n";
    std::cout << "Number of rows: " << currentRow << ", Number of cols: " << numberOfExploredStates << "\n";
    //Eigen::SparseMatrix<ValueType> mat(currentRow , numberOfExploredStates);
    //mat.setFromTriplets(tripletList.begin(), tripletList.end());
    //std::cout << mat.toDense() << std::endl;
    //spModelBuilder.constructTransitionMatrixFromTriplet(tripletList);
    spModelBuilder.constructMatrixFromTriplet(
        currentRow, numberOfExploredStates,tripletList,
        mopmc::sparse::MatrixType::Transition);
}

template<typename ValueType, typename StateType>
storm::models::sparse::StateLabeling mopmc::ExplicitModelBuilder<ValueType, StateType>::buildStateLabelling() {
    std::cout << "States in stateStorage: " << stateStorage.getNumberOfStates() << std::endl;
    return generator ->label(stateStorage, stateStorage.initialStateIndices, stateStorage.deadlockStateIndices);
}

/// Only supports BFS
template<typename ValueType, typename StateType>
StateType mopmc::ExplicitModelBuilder<ValueType, StateType>::getOrAddStateIndex(mopmc::CompressedState const& state) {
    StateType newIndex = static_cast<StateType>(stateStorage.getNumberOfStates());

    // Check if the state was already registered
    std::pair<StateType, std::size_t> actualIndexBucketPair = stateStorage.stateToId.findOrAddAndGetBucket(state, newIndex);

    StateType actualIndex = actualIndexBucketPair.first;

    if (actualIndex == newIndex) {
        statesToExplore.emplace_back(state, actualIndex);
    }

    return actualIndex;
}
