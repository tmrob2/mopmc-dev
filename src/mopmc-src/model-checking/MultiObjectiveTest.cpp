//
// Created by thomas on 20/08/23.
//
#include <iostream>
#include <memory>
#include <storm/modelchecker/multiobjective/multiObjectiveModelChecking.h>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
//#include <storm/modelchecker/multiobjective/multiObjectiveModelChecking.h>
#include "MultiObjectiveTest.h"
#include "MultiObjectivePreprocessor.h"
#include "mopmc-src/SparseModel.h"
#include <storm/models/sparse/Mdp.h>

namespace mopmc {
    namespace stormtest {

        template<typename SparseModelType>
//std::unique_ptr<storm::modelchecker::CheckResult>
        void performMultiObjectiveModelChecking(
                storm::Environment env,
                SparseModelType const &model,
                storm::logic::MultiObjectiveFormula const &formula,
                mopmc::sparse::SparseModelBuilder<double> &spModel) {

            mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::preprocess(
                env,
                model,
                formula
                //spModel
            );
        }

        template void mopmc::stormtest::performMultiObjectiveModelChecking<>(
            storm::Environment env, const storm::models::sparse::Mdp<double> &model,
            const storm::logic::MultiObjectiveFormula &formula
            //mopmc::sparse::SparseModelBuilder<double> &spModel
            );
    }
}
