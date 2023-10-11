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
#include <storm/settings/modules/GeneralSettings.h>
#include <filesystem>


bool mopmc::stormCheck(std::string const& path_to_model, std::string const& property_string) {
    storm::Environment env;
    env.modelchecker().multi().setMethod(storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa);

    //auto program = storm::parser::PrismParser::parse(path_to_model);
    storm::prism::Program program = storm::api::parseProgram(path_to_model);
    program = storm::utility::prism::preprocess(program, "");
    //std::cout << "Model Type: " << program.getModelType() << std::endl;
    // Then parse the properties, passing the program to give context to some potential variables.
    auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
    // Translate properties into the more low-level formulae.
    auto formulas = storm::api::extractFormulasFromProperties(properties);
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp =
        storm::api::buildSparseModel<double>(program, formulas)->as<storm::models::sparse::Mdp<double>>();
    uint_fast64_t const initState = *mdp->getInitialStates().begin();
    //mopmc::stormtest::performMultiObjectiveModelChecking(env, *mdp, formulas[0]->asMultiObjectiveFormula());

    std::unique_ptr<storm::modelchecker::CheckResult> result =
            storm::modelchecker::multiobjective::performMultiObjectiveModelChecking(env, *mdp, formulas[0]->asMultiObjectiveFormula());
    assert(result->isExplicitQualitativeCheckResult());
    std::cout << "Result: " << result->asExplicitQualitativeCheckResult()[initState] << std::endl;
    return true;
}


