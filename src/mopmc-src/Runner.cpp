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
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include "queries/ConvQuery.h"
#include "./model-checking/MOPMCModelChecking.h"
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include "Preprocessing.h"


namespace mopmc {


    // typedef
    typedef storm::models::sparse::Mdp<double> ModelType;
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    //typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType PrepReturnType;
    //typedef Eigen::SparseMatrix<typename ModelType::ValueType, Eigen::RowMajor> EigenSpMatrix;

    bool run(std::string const &path_to_model, std::string const &property_string) {
        //std::cout << "AN NEW ROUTINE" << std::endl;
        storm::Environment env;
        //TODO to make this work
        //auto data1 = mopmc::preprocess<ModelType>(path_to_model, property_string, env);
        //return true;

        env.modelchecker().multi().setMethod(storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa);
        //auto program = storm::parser::PrismParser::parse(path_to_model);
        storm::prism::Program program = storm::api::parseProgram(path_to_model);
        program = storm::utility::prism::preprocess(program, "");
        //std::cout << "Model Type: " << program.getModelType() << std::endl;
        // Then parse the properties, passing the program to give context to some potential variables.
        auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
        // Translate properties into the more low-level formulae.
        auto formulas = storm::api::extractFormulasFromProperties(properties);

        std::shared_ptr<storm::models::sparse::Mdp<ModelType::ValueType>> mdp =
                storm::api::buildSparseModel<ModelType::ValueType>(program, formulas)->as<ModelType>();

        std::cout << "Number of states in original mdp: " << mdp->getNumberOfStates() << "\n";
        std::cout << "Number of choices in original mdp: " << mdp->getNumberOfChoices() << "\n";

        const auto formula = formulas[0]->asMultiObjectiveFormula();
        PreprocessedType::ReturnType prepResult =
                PreprocessedType::preprocess(env, *mdp, formula);

        auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<ModelType>::analyze(prepResult);
        std::string s1 = rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::AllFinite ? "yes" : "no";
        std::string s2 = rewardAnalysis.rewardFinitenessType == storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::ExistsParetoFinite ? "yes" : "no";
        std::cout << "[!] The expected reward is finite for all objectives and all schedulers: " << s1 << std::endl;
        std::cout << "[!] There is a Pareto optimal scheduler yielding finite rewards for all objectives: " << s2 << std::endl;


        //std::ostream &outputStream = std::cout;
        //prepResult.preprocessedModel->printModelInformationToStream(outputStream);

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

        mopmc::PreprocessedData<ModelType> data(prepResult);
        //mopmc::queries::ConvexQuery q(prepResult, env);
        mopmc::queries::ConvexQuery q(data, env);
        q.query();
        return true;

    }
}