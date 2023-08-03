//
// Created by thomas on 1/08/23.
//
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include <storm/modelchecker/results/CheckResult.h>

#include <storm/utility/initialize.h>
#include <storm/builder/RewardModelBuilder.h>
#include <storm/builder/StateAndChoiceInformationBuilder.h>

#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/builder/RewardModelBuilder.h>
#include <storm/builder/StateAndChoiceInformationBuilder.h>
#include <storm/storage/sparse/StateStorage.h>

#include "ExplicitModelBuilder.h"
#include "TransitionMatrixBuilder.h"
#include <string>

typedef storm::models::sparse::Dtmc<double> Dtmc;
typedef storm::modelchecker::SparseDtmcPrctlModelChecker<Dtmc> DtmcModelChecker;

bool mopmc::check(std::string const& path_to_model, std::string const& property_string) {
    
    // Assumes that the model is in the prism program language format and parses the program.
    auto program = storm::parser::PrismParser::parse(path_to_model);
    // Code snippet assumes a Dtmc
    assert(program.getModelType() == storm::prism::Program::ModelType::DTMC);
    // Then parse the properties, passing the program to give context to some potential variables.
    auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
    // Translate properties into the more low-level formulae.
    auto formulae = storm::api::extractFormulasFromProperties(properties);

    // Now translate the prism program into a DTMC in the sparse format.
    // Use the formulae to add the correct labelling.
    //auto model = storm::api::buildSparseModel<double>(program, formulae)->template as<Dtmc>();

    // My experiments
    storm::builder::BuilderOptions options(formulae, program);
    std::shared_ptr<storm::generator::NextStateGenerator<double>> generator;
    generator = std::make_shared<storm::generator::PrismNextStateGenerator<double>>(program, options);

    mopmc::ExplicitModelBuilder<double> model(generator);
    bool deterministicModel = generator -> isDeterministicModel();
    std::vector<storm::builder::RewardModelBuilder<double>> rewardModelBuilders;

    
    std::cout << generator -> getNumberOfRewardModels() << std::endl;
    SparseMatrixBuilder transitionMatrixBuilder(0, 0, 0, false, !deterministicModel, 0);

    storm::builder::StateAndChoiceInformationBuilder stateAndChoiceInformationBuilder;
    stateAndChoiceInformationBuilder.setBuildChoiceLabels(generator->getOptions().isBuildChoiceLabelsSet());
    stateAndChoiceInformationBuilder.setBuildChoiceOrigins(generator->getOptions().isBuildChoiceOriginsSet());
    
    std::cout << generator -> getOptions().isBuildStateValuationsSet() << std::endl;
    stateAndChoiceInformationBuilder.setBuildStateValuations(generator->getOptions().isBuildStateValuationsSet());

    model.buildMatrices(transitionMatrixBuilder, rewardModelBuilders, stateAndChoiceInformationBuilder);
    // Ok now that we have these three we can start investigating the build matrices routine
    //Create a callback for the nest-state generator to enable it to request the index of the states

    // Create a model checker on top of the sparse engine.
    //auto checker = std::make_shared<DtmcModelChecker>(*model);
    // Create a check task with the formula. Run this task with the model checker.
    //auto result = checker->check(storm::modelchecker::CheckTask<>(*(formulae[0]), true));
    //assert(result->isExplicitQuantitativeCheckResult());
    // Use that we know that the model checker produces an explicit quantitative result
    //auto quantRes = result->asExplicitQuantitativeCheckResult<double>();
    // Now compare the result at the first initial state of the model with 0.5.
    //return quantRes[*model->getInitialStates().begin()] > 0.5;
    
    return true;
}

template<typename ValueType, typename RewardModelType, typename StateType>
void mopmc::ExplicitModelBuilder<ValueType, RewardModelType, StateType>::buildMatrices(
    SparseMatrixBuilder& transitionMatrixBuilder,
    std::vector<storm::builder::RewardModelBuilder<typename RewardModelType::ValueType>>& rewardModelBuilders,
    storm::builder::StateAndChoiceInformationBuilder& stateAndChoiceInformationBuilder
) {
    // Create a callback function for the next state generator to enable it to request the index of states
    std::function<StateType(CompressedState const&)> stateToIdCallback = 
        std::bind(&ExplicitModelBuilder<ValueType, RewardModelType, StateType>::getOrAddStateIndex, this, std::placeholders::_1);

    // Assume that the exploration order is BFS
    this->stateStorage.initialStateIndices = generator->getInitialStates(stateToIdCallback);
    std::cout << "Initial states: " << "\n";
    for(uint32_t state : stateStorage.initialStateIndices) {
        std::cout << state << std::endl;
    }

    // Explore the current state until there are no more reachable states
    uint_fast64_t currentRowGroup = 0;
    uint_fast64_t currentRow = 0;

    uint64_t numberOfExploredStates = 0;
    uint64_t numberOfExploredStatesSinceLastMessage = 0;

    // Perform the search through the model
    
    // first just get the next states
    mopmc::CompressedState currentState = statesToExplore.front().first;
    StateType currentIndex = statesToExplore.front().second;
    statesToExplore.pop_front();

    std::cout << "state: " << currentState << ",idx: " << currentIndex << std::endl;
}

/// Only supports BFS
template<typename ValueType, typename RewardModelType, typename StateType>
StateType mopmc::ExplicitModelBuilder<ValueType, RewardModelType, StateType>::getOrAddStateIndex(mopmc::CompressedState const& state) {
    StateType newIndex = static_cast<StateType>(stateStorage.getNumberOfStates());

    // Check if the state was already registered
    std::pair<StateType, std::size_t> actualIndexBucketPair = stateStorage.stateToId.findOrAddAndGetBucket(state, newIndex);

    StateType actualIndex = actualIndexBucketPair.first;

    statesToExplore.emplace_back(state, actualIndex);

    return actualIndex;
}
