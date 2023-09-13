//
// Created by thomas on 20/08/23.
//
#include <iostream>
#include <memory>
#include <storm/modelchecker/multiobjective/multiObjectiveModelChecking.h>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/modelchecker/multiobjective/pcaa/SparsePcaaQuery.h>
#include <storm/modelchecker/multiobjective/pcaa/SparsePcaaAchievabilityQuery.h>
#include "SparseMultiObjective.h"
#include "MultiObjectivePreprocessor.h"
#include <storm/models/sparse/Mdp.h>
#include "StandardMdpPcaaChecker.h"

namespace mopmc {
namespace stormtest {

template<typename SparseModelType>
//std::unique_ptr<storm::modelchecker::CheckResult>
void performMultiObjectiveModelChecking(
        storm::Environment env,
        SparseModelType &model,
        storm::logic::MultiObjectiveFormula const &formula) {

    typename mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::ReturnType result =
            mopmc::stormtest::SparseMultiObjectivePreprocessor<SparseModelType>::preprocess(
                env,
                model,
                formula);

    // ok now we start experimenting with the result
    std::ostream& outputStream = std::cout;

    result.preprocessedModel->printModelInformationToStream(outputStream);

    storm::modelchecker::multiobjective::MultiObjectiveMethod method = env.modelchecker().multi().getMethod();
    switch(method) {
        case storm::modelchecker::multiobjective::MultiObjectiveMethod::Pcaa:
            std::cout << "Pcaa\n";
            break;
        default:
            std::cout << "Some other multi-objective method";
    }

    std::cout << "Trivial objectives: " << (result.containsOnlyTrivialObjectives() ? "yes" : "no") << "\n";

    mopmc::multiobjective::StandardMdpPcaaChecker<SparseModelType> mdpChecker(result);

    mdpChecker.multiObjectiveSolver(env);

}

// Explicit

template void mopmc::stormtest::performMultiObjectiveModelChecking(
    storm::Environment env, storm::models::sparse::Mdp<double> &model,
    const storm::logic::MultiObjectiveFormula &formula);
}
}
