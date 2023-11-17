//
// Created by guoxin on 17/11/23.
//

#include "Preprocessor.h"
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

    typedef storm::models::sparse::Mdp<double> ModelType;
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType PrepReturnType;

    template<typename M>
    PreprocessedData<M>::PreprocessedData() {}


    template<typename M>
    PreprocessedData<M>::PreprocessedData(
            typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn) {

        mopmc::multiobjective::MOPMCModelChecking<M> model(prepReturn);

        transitionMatrix = *storm::adapters::EigenAdapter::toEigenSparseMatrix(model.getTransitionMatrix());
        transitionMatrix.makeCompressed();
        rowCount = model.getTransitionMatrix().getRowCount();
        colCount = model.getTransitionMatrix().getColumnCount();
        rowGroupIndices = model.getTransitionMatrix().getRowGroupIndices();
        row2RowGroupMapping.resize(rowCount);

        for (uint64_t i = 0; i < rowGroupIndices.size()-1; ++i) {
            size_t currInd = rowGroupIndices[i];
            size_t nextInd = rowGroupIndices[i + 1];
            for (uint64_t j = 0; j < nextInd - currInd; ++j)
                row2RowGroupMapping[currInd + j] = i;
        }
        numObjs = prepReturn.objectives.size();
        rewardVectors = model.getActionRewards();
        assert(rewardVectors.size() == numObjs);
        assert(rewardVectors[0].size() == rowCount);
        flattenRewardVector.resize(numObjs*rowCount);
        for (uint64_t i = 0; i < numObjs; ++i) {
            for (uint_fast64_t j = 0; j < rowCount; ++j) {
                flattenRewardVector[i * rowCount + j] = rewardVectors[i][j];
            }
        }
        thresholds.resize(numObjs);
        isProbObj.resize(numObjs);
        for (uint_fast64_t i = 0; i < numObjs; ++i) {
            thresholds[i] = prepReturn.objectives[i].formula->template getThresholdAs<typename M::ValueType>();
            isProbObj[i] = prepReturn.objectives[i].originalFormula->isProbabilityOperatorFormula();
        }

        defaultScheduler.assign(colCount, static_cast<uint64_t>(0));
        iniRow = model.getInitialState();

    }

    template<typename M>
    PreprocessedData<M> preprocess(std::string const &path_to_model, std::string const &property_string, storm::Environment env) {

        //auto program = storm::parser::PrismParser::parse(path_to_model);
        storm::prism::Program program = storm::api::parseProgram(path_to_model);
        program = storm::utility::prism::preprocess(program, "");
        //std::cout << "Model Type: " << program.getModelType() << std::endl;
        // Then parse the properties, passing the program to give context to some potential variables.
        auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
        // Translate properties into the more low-level formulae.
        auto formulas = storm::api::extractFormulasFromProperties(properties);

        const auto formula = formulas[0]->asMultiObjectiveFormula();

        std::shared_ptr<storm::models::sparse::Mdp<typename M::ValueType>> mdp =
                storm::api::buildSparseModel<typename M::ValueType>(program, formulas)->template as<M>();
        //std::cout << "Num of states in parsed MDP: " << mdp->getNumberOfStates() << "\n";
        //std::cout << "Num of choices in parsed MDP: " << mdp->getNumberOfChoices() << "\n";
        PreprocessedType::ReturnType prepResult =
                PreprocessedType::preprocess(env, *mdp, formula);

        auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<ModelType>::analyze(
                prepResult);
        std::string s1 = rewardAnalysis.rewardFinitenessType ==
                         storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::AllFinite ? "yes"
                                                                                                             : "no";
        std::string s2 = rewardAnalysis.rewardFinitenessType ==
                         storm::modelchecker::multiobjective::preprocessing::RewardFinitenessType::ExistsParetoFinite
                         ? "yes" : "no";
        std::cout << "[!] The expected reward is finite for all objectives and all schedulers: " << s1 << std::endl;
        std::cout << "[!] There is a Pareto optimal scheduler yielding finite rewards for all objectives: " << s2
                  << std::endl;

        std::ostream &outputStream = std::cout;
        //prepResult.preprocessedModel->printModelInformationToStream(outputStream);

        //Objectives must be total rewards
        if (!prepResult.containsOnlyTotalRewardFormulas()) {
            throw std::runtime_error("This framework handles total rewards only.");
        }
        //Confine the property (syntax) to achievability query
        // We will convert it to a convex query.
        if (prepResult.queryType != storm::modelchecker::multiobjective::preprocessing
        ::SparseMultiObjectivePreprocessorResult<ModelType>::QueryType::Achievability) {
            throw std::runtime_error("the input property should be achievability query type.");
        }


        return mopmc::PreprocessedData<M>(prepResult);
    };

    template class PreprocessedData<ModelType>;
    //void preprocess(std::string const &path_to_model, std::string const &property_string, storm::Environment env);
    template PreprocessedData<ModelType> preprocess(std::string const &path_to_model, std::string const &property_string, storm::Environment env);
}
