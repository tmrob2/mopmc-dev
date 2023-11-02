//
// Created by guoxin on 2/11/23.
//

#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include "ExplicitModelBuilder.h"
//#include "model-checking/SparseMultiObjective.h"
#include <string>
#include <iostream>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>

#include "Runner.h"
#include "model-checking/MultiObjectivePreprocessor.h"
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
//#include <storm/modelchecker/multiobjective/pcaa/StandardPcaaWeightVectorChecker.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include <storm/storage/BitVector.h>

bool mopmc::run(std::string const &path_to_model, std::string const &property_string) {
    //std::cout << "AN NEW ROUTINE" << std::endl;
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

    typedef storm::models::sparse::Mdp<double> ModelType;
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp =
            storm::api::buildSparseModel<double>(program, formulas)->as<ModelType>();

    const auto formula = formulas[0]->asMultiObjectiveFormula();

    /*
    typename mopmc::multiobjective::SparseMultiObjectivePreprocessor<ModelType>::ReturnType result =
            mopmc::multiobjective::SparseMultiObjectivePreprocessor<ModelType>::preprocess(
                    env,
                    *mdp,
                    formula);
    */
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    typename PreprocessedType::ReturnType prepResult = PreprocessedType::preprocess(
            env,
            *mdp,
            formula);

    std::ostream& outputStream = std::cout;

    prepResult.preprocessedModel->printModelInformationToStream(outputStream);

    uint_fast64_t numOfRows = prepResult.preprocessedModel->getTransitionMatrix().getRowCount();
    std::vector<typename ModelType::ValueType>
            weightedRewardVector(numOfRows, storm::utility::zero<typename ModelType::ValueType>());

    //Objectives must be total rewards
    for (auto &objective : prepResult.objectives) {
        if (!objective.formula->getSubformula().isTotalRewardFormula()) {
            throw std::runtime_error("This framework handles total rewards only");
        }
    }

    storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<ModelType> stormModelChecker(prepResult);
    //std::cout << "AFTER INITIALISATION" << std::endl;
    //prepResult.preprocessedModel->printModelInformationToStream(outputStream);

    // TODO
    //It calls a query (Alg. 1) in ./mompc-src/queries...

    return true;
}