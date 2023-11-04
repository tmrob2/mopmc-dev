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
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include "../solvers/ConvexQuery.h"


namespace mopmc::queries {

    typedef typename ModelType::ValueType T;

    ConvexQuery::ConvexQuery(const PrepReturnType& t) : t_(t){}

    ConvexQuery::ConvexQuery(const PrepReturnType& t,
                             const storm::Environment& env) :
                             t_(t), env_(env){}

    void ConvexQuery::query() {

        //Data generation
        const uint64_t m = t_.objectives.size(); // m: number of objectives

        std::vector<std::vector<T>> rho(m); //rho: all reward vectors
        for (uint_fast64_t i = 0; i < m; i++) {
            auto &name_ = t_.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
            rho[i] = t_.preprocessedModel->getRewardModel(name_)
                    .getTotalRewardVector(t_.preprocessedModel->getTransitionMatrix());
        }

        auto P = // P: transition matrix as eigen sparse matrix
                storm::adapters::EigenAdapter::toEigenSparseMatrix(t_.preprocessedModel->getTransitionMatrix());
        P->makeCompressed();

        //Initialisation
        std::vector<std::vector<T>> Phi;
        // Lam1, Lam2 represent Lambda
        std::vector<std::vector<T>> Lam1;
        std::vector<std::vector<T>> Lam2;

        std::vector<T> h(m); //h: thresholds in objectives
        for (uint_fast64_t i = 0; i < m; i++) {
            h[i] = t_.objectives[i].formula->getThresholdAs<T>();
        }
        std::vector<T> vt = std::vector<T>(m, static_cast<T>(1.0));
        std::vector<T> vb = std::vector<T>(m, static_cast<T>(1.0));

        std::vector<ModelType::ValueType> w = // w: weight vector
                std::vector<T>(m, static_cast<T>(1.0) / static_cast<T>(m));

        //thresholds for stopping the iteration
        const double eps{0.001};
        const double eps1{0.001};
        const uint_fast64_t maxIter{10};

        //Iteration
        uint_fast64_t iter = 0;
        T fDiff = 0;
        while (iter < maxIter && (Phi.size() < 3 || fDiff > eps)) {
            iter ++;
            //std::cout << "Iteration: " << iter << "\n";
            std::vector<T> fvt = mopmc::solver::convex::ReLU(vt, h);
            std::vector<T> fvb = mopmc::solver::convex::ReLU(vb, h);
            fDiff = mopmc::solver::convex::diff(fvt, fvb);
            if (Phi.size() > 0) {
                continue;
                //TODO
                //vt = argmin ...
                if (true) {
                    continue;
                        }
            }
        }

        //call an existing solver for development purposes
        //intend to migrate the solver's code to the current file...
        mopmc::multiobjective::MOPMCModelChecking<ModelType> testChecker(t_);
        //storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<ModelType> testChecker(t);
        //testChecker.check(env_, w);
        testChecker.multiObjectiveSolver(env_);
        std::cout << "To implement the convex query ... \n";
    }

}
