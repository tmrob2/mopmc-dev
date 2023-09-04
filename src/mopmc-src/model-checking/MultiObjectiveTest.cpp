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
#include <storm/models/sparse/Mdp.h>
#include "../SparseModel2.h"
#include "../solvers/OptimalPolicy.h"

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

            // ok so the next thing to do is implement value iteration using sparse matrices to solve this system
            mopmc::sparsemodel::SparseModelBuilder<SparseModelType> spModel(result);

            // construct a problem
            std::vector<typename SparseModelType::ValueType>
                    x_(spModel.getTransitionMatrix().cols(), static_cast<typename SparseModelType::ValueType>(0.0));
            std::vector<typename SparseModelType::ValueType>
                    y_(spModel.getTransitionMatrix().rows(), static_cast<typename SparseModelType::ValueType>(0.0));
            Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> x(x_.data(), x_.size());
            Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> y(y_.data(), y_.size());
            std::vector<uint_fast64_t> pi(spModel.getNumberOfStates());

            std::cout << "Number of reward models " << spModel.getRewardModelNames().size() << "\n";
            double initVal = 1.0 / static_cast<double>(spModel.getRewardModelNames().size());
            std::vector<typename SparseModelType::ValueType>
                    w_(spModel.getRewardModelNames().size(), static_cast<typename SparseModelType::ValueType>(initVal));
            Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> w(w_.data(), w_.size());

            mopmc::solver::Problem<SparseModelType> problem(
                    spModel,static_cast<typename SparseModelType::ValueType>(0.0001), x, y, pi,
                    result.objectives[0].formula->getOptimalityType());
            mopmc::solver::optimalPolicy(problem, w);
        }

        template void mopmc::stormtest::performMultiObjectiveModelChecking(
            storm::Environment env, storm::models::sparse::Mdp<double> &model,
            const storm::logic::MultiObjectiveFormula &formula);
    }
}
