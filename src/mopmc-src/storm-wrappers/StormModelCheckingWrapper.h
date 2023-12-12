//
// Created by guoxin on 1/12/23.
//

#ifndef MOPMC_STORMMODELCHECKINGWRAPPER_H
#define MOPMC_STORMMODELCHECKINGWRAPPER_H

#include "storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h"
#include "storm/modelchecker/multiobjective/constraintbased/SparseCbAchievabilityQuery.h"
#include "storm/modelchecker/multiobjective/deterministicScheds/DeterministicSchedsParetoExplorer.h"
#include "storm/modelchecker/multiobjective/pcaa/SparsePcaaAchievabilityQuery.h"
#include "storm/modelchecker/multiobjective/pcaa/SparsePcaaParetoQuery.h"
#include "storm/modelchecker/multiobjective/pcaa/SparsePcaaQuantitativeQuery.h"
#include "storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h"
#include "storm/models/sparse/MarkovAutomaton.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/settings/SettingsManager.h"
#include "storm/settings/modules/CoreSettings.h"
#include "storm/utility/macros.h"

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/InvalidEnvironmentException.h"

//TODO currently this is not working...
namespace mopmc::wrapper {


    template<typename ModelType>
    class StormModelCheckingWrapper {
    public:

        explicit StormModelCheckingWrapper(
                typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType &preprocessorResult
        ) : preprocessorResult_(preprocessorResult) {}

        //void performMultiObjectiveModelChecking(storm::Environment const &env, ModelType const &model,
        //                                        storm::logic::MultiObjectiveFormula const &formula);
        void performMultiObjectiveModelChecking(storm::Environment const &env);

        typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType preprocessorResult_;
        std::unique_ptr<storm::modelchecker::CheckResult> result;

    };

    template<typename ModelType>
    void StormModelCheckingWrapper<ModelType>::performMultiObjectiveModelChecking(const storm::Environment &env) {

        std::unique_ptr<storm::modelchecker::multiobjective::SparsePcaaQuery<ModelType, storm::RationalNumber>> query;

        query = std::unique_ptr<storm::modelchecker::multiobjective::SparsePcaaQuery<ModelType, storm::RationalNumber>>(
                new typename storm::modelchecker::multiobjective::SparsePcaaAchievabilityQuery<ModelType, storm::RationalNumber>(
                        this->preprocessorResult_));

        if (query) {
            std::cout << "NON NULL\n";
        } else
        {std::cout << "NON NULL\n";}



        std::cout << "get here.\n";
        //this->result = query->check(env);
        std::cout << "get here?\n";
    }

    template
    class StormModelCheckingWrapper<storm::models::sparse::Mdp<double>>;
}

#endif //MOPMC_STORMMODELCHECKINGWRAPPER_H
