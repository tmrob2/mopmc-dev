//
// Created by guoxin on 3/11/23.
//


#include <iostream>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <storm/adapters/EigenAdapter.h>
#include "ConvQuery.h"
#include <storm/api/storm.h>
#include "../model-checking/SparseMultiObjective.h"
#include "../model-checking/MOPMCModelChecking.h"


namespace mopmc::queries {

    typedef typename ModelType::ValueType T;

    ConvexQuery::ConvexQuery(const PrepReturnType& t_) : t(t_){}

    ConvexQuery::ConvexQuery(const PrepReturnType& t_,
                             const storm::Environment& env_) :
                             t(t_), env(env_){}

    void ConvexQuery::query() {

        //Data generation
        uint64_t m = t.objectives.size(); // number of objectives

        std::vector<std::vector<T>> rvs(m); //all reward vectors
        for (uint_fast64_t i = 0; i < m; i++) {
            auto &name_ = t.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
            rvs[i] = t.preprocessedModel->getRewardModel(name_)
                    .getTotalRewardVector(t.preprocessedModel->getTransitionMatrix());
        }

        auto M = // transition matrix as eigen sparse matrix
                storm::adapters::EigenAdapter::toEigenSparseMatrix(t.preprocessedModel->getTransitionMatrix());
        M->makeCompressed();

        //Initialisation
        std::vector<std::vector<T>> Phi;
        // Lam1, Lam2 represent Lambda
        std::vector<std::vector<T>> Lam1;
        std::vector<std::vector<T>> Lam2;

        std::vector<T> h(m); //thresholds in objectives
        for (uint_fast64_t i = 0; i < m; i++) {
            h[i] = t.objectives[i].formula->getThresholdAs<T>();
        }
        std::vector<T> vt = std::vector<T>(m, static_cast<T>(1.0));
        std::vector<T> vd = std::vector<T>(m, static_cast<T>(1.0));

        std::vector<ModelType::ValueType> w = // (initial) weight vector
                std::vector<T>(m, static_cast<T>(1.0) / static_cast<T>(m));

        const double eps{0.001};
        const double eps1{0.001};
        const uint_fast64_t maxIter{10};

        //Iteration
        uint_fast64_t iter = 0;
        while (iter < maxIter && Phi.size() < 3) {
            iter ++;
            std::cout << "Iter: " << iter << std::endl;
        }
        mopmc::multiobjective::MOPMCModelChecking<ModelType> testChecker(t);
        testChecker.check(env, w);
        std::cout << "To implement the convex query ... \n";
    }

}
