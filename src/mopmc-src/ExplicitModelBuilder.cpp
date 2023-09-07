//
// Created by thomas on 1/08/23.
//
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include "ExplicitModelBuilder.h"
#include "model-checking/MultiObjectiveTest.h"
#include <string>
#include <iostream>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>


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


