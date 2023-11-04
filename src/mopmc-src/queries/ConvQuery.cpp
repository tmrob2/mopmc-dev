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

    ConvexQuery::ConvexQuery(const PrepReturnType &t) : t_(t) {}

    ConvexQuery::ConvexQuery(const PrepReturnType &t,
                             const storm::Environment &env) :
            t_(t), env_(env) {}

    void ConvexQuery::query() {

        //Data generation
        const uint64_t m = t_.objectives.size(); // m: number of objectives

        std::vector <std::vector<T>> rho(m); //rho: all reward vectors
        for (uint_fast64_t i = 0; i < m; i++) {
            auto &name_ = t_.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
            rho[i] = t_.preprocessedModel->getRewardModel(name_)
                    .getTotalRewardVector(t_.preprocessedModel->getTransitionMatrix());
        }

        auto P = // P: transition matrix as eigen sparse matrix
                storm::adapters::EigenAdapter::toEigenSparseMatrix(t_.preprocessedModel->getTransitionMatrix());
        P->makeCompressed();

        //Initialisation
        std::vector <std::vector<T>> Phi;
        // Lam1, Lam2 represent Lambda
        std::vector <std::vector<T>> Lam1;
        std::vector <std::vector<T>> Lam2;

        std::vector <T> h(m); //h: thresholds in objectives
        for (uint_fast64_t i = 0; i < m; i++) {
            h[i] = t_.objectives[i].formula->getThresholdAs<T>();
        }
        Eigen::Map <Eigen::Matrix<T, Eigen::Dynamic, 1>> h_(h.data(), h.size());

        //vt, vb
        std::vector <T> vt = std::vector<T>(m, static_cast<T>(0.0));
        std::vector <T> vb = std::vector<T>(m, static_cast<T>(0.0));
        //vi: initial vector for Frank-Wolfe
        std::vector <T> vi;

        std::vector <ModelType::ValueType> w = // w: weight vector
                std::vector<T>(m, static_cast<T>(1.0) / static_cast<T>(m));

        //thresholds for stopping the iteration
        const double eps{0.001};
        const double eps1{0.001};
        const uint_fast64_t maxIter{10};

        //GS: Double-check, from an algorithmic and practical point of view,
        // whether we maintain the two data structures
        // in the main iteration below. :SG
        std::vector <std::vector<T>> W;
        std::set <std::vector<T>> wSet;

        //GS:
        // :SG
        mopmc::multiobjective::MOPMCModelChecking<ModelType> testChecker(t_);
        //storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<ModelType> testChecker(t);

        //Iteration
        uint_fast64_t iter = 0;
        T fDiff = 0;
        while (iter < maxIter && (Phi.size() < 3 || fDiff > eps)) {
            iter++;
            //std::cout << "Iteration: " << iter << "\n";
            std::vector <T> fvt = mopmc::solver::convex::ReLU(vt, h);
            std::vector <T> fvb = mopmc::solver::convex::ReLU(vb, h);
            fDiff = mopmc::solver::convex::diff(fvt, fvb);
            if (Phi.size() > 0) {
                //GS:
                if (Phi.size() == 1) {
                    vi = Phi.back();
                }

                //GS: Here why do we need to make the argument of reluGradient
                // an Eigen matrix rather than a standard one
                // (like in ReLU)? :SG
                // compute the FW and find a new weight vector
                vt = mopmc::solver::convex::frankWolfe(mopmc::solver::convex::reluGradient<T>,
                                                       vi, 100, W, Phi, h);
                Eigen::Map <Eigen::Matrix<T, Eigen::Dynamic, 1>> vt_(vt.data(), vt.size());
                Eigen::Matrix<T, Eigen::Dynamic, 1> cx = h_ - vt_;
                std::vector <T> grad = mopmc::solver::convex::reluGradient(cx);
                // If a w has already been seen before break;
                // make sure we call it w
                w = mopmc::solver::convex::computeNewW(grad);
            }
            /*
            std::cout << "w*: ";
            for (int i = 0; i < w.size() ; ++i ){
                std::cout << w[i] << ",";
            }
            std::cout << "\n";
             */

            //GS: As mention, double check whether we need to
            // maintain W and wSet. :SG
            // if the w generated is already contained with W
            if (wSet.find(w) != wSet.end()) {
                std::cout << "W already in set => W ";
                for (auto val: w) {
                    std::cout << val << ", ";
                }
                std::cout << "\n";
                break;
            } else {
                W.push_back(w);
                wSet.insert(w);
            }

            // compute a new supporting hyperplane
            std::cout << "W[" << iter << "]: ";
            for (T val: W.back()) {
                std::cout << val << ",";
            }
            std::cout << "\n";


            //testChecker.check(env_, w);
        }


        //testChecker.check(env_, w);
        //testChecker.multiObjectiveSolver(env_);
        std::cout << "To implement the convex query ... \n";
    }

}
