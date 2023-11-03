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
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
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
    std::shared_ptr<storm::models::sparse::Mdp<ModelType::ValueType>> mdp =
            storm::api::buildSparseModel<ModelType::ValueType>(program, formulas)->as<ModelType>();

    const auto formula = formulas[0]->asMultiObjectiveFormula();

    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    typename PreprocessedType::ReturnType prepResult =
            PreprocessedType::preprocess(env, *mdp, formula);

    std::ostream &outputStream = std::cout;
    prepResult.preprocessedModel->printModelInformationToStream(outputStream);

    //Objectives must be total rewards
    if (!prepResult.containsOnlyTotalRewardFormulas()) {
        throw std::runtime_error("This framework handles total rewards only.");
    }
    //Confine the property (syntax) to achievability query
    // We will convert it to a convex query.
    if (prepResult.queryType != storm::modelchecker::multiobjective::preprocessing
    ::SparseMultiObjectivePreprocessorResult<ModelType>::QueryType::Achievability) {
        throw std::runtime_error("The input property should be achievability query type.");
    }

    //Get thresholds
    uint_fast64_t numOfObjs = prepResult.objectives.size();
    std::vector<ModelType::ValueType> thresholds(numOfObjs);
    std::cout << "The thresholds are: ";
    for (uint_fast64_t i=0; i<numOfObjs; i++) {
        auto thres = prepResult.objectives[i].formula->getThresholdAs<double>();
        thresholds[i] = thres;
        std::cout << thres << ", ";
    }
    std::cout << std::endl;

    //Initialise the model
    // Because initialize() is protected, need to create a mocked model checker object
    // whose constructor calls initialize().
    // For our purposes, we use this function to check reward finiteness
    // and other desirable requirements of the model.
    try {
        new storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<ModelType>(prepResult);
    }
    catch (const std::runtime_error& e){ std::cout << e.what() << "\n";}
    //prepResult.preprocessedModel->printModelInformationToStream(outputStream);

    //Generate reward vectors
    std::vector<std::vector<ModelType::ValueType>> rewVectors(numOfObjs);
    for (uint_fast64_t i=0; i<numOfObjs; i++) {
        const auto& rewModelName = prepResult.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
        //std::cout << "Reward model name: " << rewModelName << ", Reward model type: "
        //    << prepResult.preprocessedModel->getRewardModel(rewModelName) << std::endl;
        rewVectors[i] = prepResult.preprocessedModel->getRewardModel(rewModelName)
                .getTotalRewardVector(prepResult.preprocessedModel->getTransitionMatrix());
        // for (const auto& x: rewVectors[i]) std::cout << x << ' ';
    }

    //Initialise weight vector
    std::vector<ModelType::ValueType> weightVector(numOfObjs);
    //Initialise weighted reward vector
    uint64_t numOfRows = prepResult.preprocessedModel->getTransitionMatrix().getRowCount();
    std::vector<ModelType::ValueType> wRewVector(numOfRows);
    //Initialise scheduler
    uint64_t numOfRowGroups = prepResult.preprocessedModel->getTransitionMatrix().getRowGroupCount();
    std::vector<uint64_t > scheduler(numOfRowGroups);
    //Convert transition matrix to eigen sparse matrix
    // TODO

    //It calls a query (Alg. 1) in ./mompc-src/queries...
    // TODO

    return true;
}