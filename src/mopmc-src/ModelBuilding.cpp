//
// Created by guoxin on 17/11/23.
//

#include "ModelBuilding.h"
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm/storage/prism/Program.h>
#include <string>
#include <iostream>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/environment/Environment.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include "model-checking/MOPMCModelChecking.h"

namespace mopmc {

    //typedef storm::models::sparse::Mdp<double> ModelType;
    //typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    //typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType PrepReturnType;

    template<typename M>
    typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType
    ModelBuilder<M>::build(const std::string &path_to_model, const std::string &property_string,
                           storm::Environment &env) {

        env.modelchecker().multi().setMethod(storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa);

        //auto program = storm::parser::PrismParser::parse(path_to_model);
        storm::prism::Program program = storm::api::parseProgram(path_to_model);
        program = storm::utility::prism::preprocess(program, "");
        //std::cout << "Model Type: " << program.getModelType() << std::endl;
        // Then parse the properties, passing the program to give context to some potential variables.
        auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
        // Translate properties into the more low-level formulae.
        auto formulas = storm::api::extractFormulasFromProperties(properties);

        std::shared_ptr<storm::models::sparse::Mdp<typename M::ValueType>> mdp =
                storm::api::buildSparseModel<typename M::ValueType>(program, formulas)->template as<M>();

        std::cout << "Number of states in original mdp: " << mdp->getNumberOfStates() << "\n";
        std::cout << "Number of choices in original mdp: " << mdp->getNumberOfChoices() << "\n";

        const auto formula = formulas[0]->asMultiObjectiveFormula();
        typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType prepResult =
                storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::preprocess(env, *mdp, formula);

        auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<M>::analyze(prepResult);
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
        ::SparseMultiObjectivePreprocessorResult<M>::QueryType::Achievability) {
            throw std::runtime_error("The input property should be achievability query type.");
        }

        return prepResult;
    }

    template class ModelBuilder<storm::models::sparse::Mdp<double>>;
}
