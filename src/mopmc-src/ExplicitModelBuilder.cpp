//
// Created by thomas on 1/08/23.
//
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include "ExplicitModelBuilder.h"
#include "model-checking/SparseMultiObjective.h"
#include <string>
#include <iostream>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/modelchecker/multiobjective/multiObjectiveModelChecking.h>
#include <storm-parsers/parser/NondeterministicSparseTransitionParser.h>
#include <storm-parsers/parser/SparseItemLabelingParser.h>
#include <storm-parsers/parser/SparseStateRewardParser.h>
#include <storm-parsers/parser/FormulaParser.h>
#include <filesystem>


bool mopmc::stormCheck(std::string const& path_to_model, std::string const& property_string) {
    storm::Environment env;
    auto program = storm::parser::PrismParser::parse(path_to_model);
    std::cout << "Model Type: " << program.getModelType() << std::endl;
    // Then parse the properties, passing the program to give context to some potential variables.
    auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
    // Translate properties into the more low-level formulae.
    auto formulas = storm::api::extractFormulasFromProperties(properties);
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp =
        storm::api::buildSparseModel<double>(program, formulas)->as<storm::models::sparse::Mdp<double>>();
    mopmc::stormtest::performMultiObjectiveModelChecking(env, *mdp, formulas[0]->asMultiObjectiveFormula());
    return true;
}

//! Parse a non-deterministic explicit custom model. Storm doesn't have the function to
//! import multiple rewards models but we still need them. So we import a custom transition function and multiple
//! rewards models based on a vector of inputs. In this case the rewards models need to be contained in a directory
//! and we input all of the rewards models in this directory
bool mopmc::stormWarehouseExperiment(std::string const& path_to_explicit_model, std::string const& path_to_explicit_labelling,
                std::vector<std::string> const& path_to_rewards_model, std::string const& propFname){
    // storm can't really handle this case so we force storm to handle multiple rewards models
    storm::Environment env;
    // parse the transitions
    storm::storage::SparseMatrix<double> transitions(
            std::move(storm::parser::NondeterministicSparseTransitionParser<double>::parseNondeterministicTransitions(path_to_explicit_model)));

    uint_fast64_t stateCount = transitions.getColumnCount();

    // parse the state labelling
    storm::models::sparse::StateLabeling labeling(
            storm::parser::SparseItemLabelingParser::parseAtomicPropositionLabeling(stateCount, path_to_explicit_labelling));

    // Initialise the result
    storm::storage::sparse::ModelComponents<double> result(std::move(transitions), std::move(labeling));

    // If the rewards vector is not empty then loop over the rewards files and insert them into the model

    std::vector<std::vector<double>> rewardModels;
    for (const auto& fname : path_to_rewards_model) {
        std::vector<double> stateRewards;
        rewardModels.push_back(
                std::move(storm::parser::SparseStateRewardParser<double>::parseSparseStateReward(stateCount, fname)));
    }

    uint count = 0;
    for (auto& rewardModel: rewardModels) {
        std::filesystem::path path(path_to_rewards_model[count]);
        std::string rewardModelName = path.filename().stem();
        std::cout << "reward model name: " << rewardModelName << std::endl;
        result.rewardModels.insert(std::make_pair(
            rewardModelName,storm::models::sparse::StandardRewardModel<double>(std::move(rewardModel), std::nullopt, std::nullopt)));
        ++count;
    }

    // the result now needs to be converted to an MDP
    storm::models::sparse::Mdp<double> mdp = storm::models::sparse::Mdp<double>(std::move(result));

    // now we are back at the stage of the original Prism parser implementation

    // parse the formula
    // open then propFile and parse the string.
    storm::parser::FormulaParser formulaParser;
    std::shared_ptr<storm::logic::Formula const> formula = formulaParser.parseSingleFormulaFromString(readPropFile(propFname));

    mopmc::stormtest::performMultiObjectiveModelChecking(env, mdp, formula->asMultiObjectiveFormula());
    return 1;
}

std::string mopmc::readPropFile(std::string const& fname) {
    std::ifstream inputFile(fname);

    if (!inputFile.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
    }

    std::string firstLine;
    if (std::getline(inputFile, firstLine)) {
        // successfully read the first line into the firstline string
        std::cout << "Read property: " << firstLine << std::endl;
    } else {
        std::cout << "File is empty!" << std::endl;
    }

    inputFile.close();
    return firstLine;
}


