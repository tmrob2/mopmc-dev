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

    ConvexQuery::ConvexQuery(const PrepReturnType &model) : model_(model) {}

    ConvexQuery::ConvexQuery(const PrepReturnType &model,
                             const storm::Environment &env) :
            model_(model), env_(env) {}

    void ConvexQuery::query() {

        //Data generation
        const uint64_t m = model_.objectives.size(); // m: number of objectives

        std::vector<std::vector<T>> rho(m); //rho: all reward vectors
        //GS: need to store whether an objective is probabilistic or reward-based.
        //TODO In future we will use treat them differently in the loss function. :GS
        std::vector<bool> isProbObj(m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            auto &name_ = model_.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
            rho[i] = model_.preprocessedModel->getRewardModel(name_)
                    .getTotalRewardVector(model_.preprocessedModel->getTransitionMatrix());
            isProbObj[i] = model_.objectives[i].originalFormula->isProbabilityOperatorFormula();
        }

        auto P = // P: transition matrix as eigen sparse matrix
                storm::adapters::EigenAdapter::toEigenSparseMatrix(model_.preprocessedModel->getTransitionMatrix());
        P->makeCompressed();

        //Initialisation
        std::vector<std::vector<T>> Phi;
        // LambdaL, LambdaR represent Lambda
        std::vector<std::vector<T>> LambdaL;
        std::vector<std::vector<T>> LambdaR;

        std::vector<T> h(m); //h: thresholds in objectives
        for (uint_fast64_t i = 0; i < m; ++i) {
            h[i] = model_.objectives[i].formula->getThresholdAs<T>();
        }
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> h_(h.data(), h.size());

        //vt, vb
        std::vector<T> vt = std::vector<T>(m, static_cast<T>(0.0));
        std::vector<T> vb = std::vector<T>(m, static_cast<T>(0.0));
        //vi: initial vector for Frank-Wolfe
        std::vector<T> *vi;
        std::vector<T> r(m);

        std::vector<ModelType::ValueType> w = // w: weight vector
                std::vector<T>(m, static_cast<T>(1.0) / static_cast<T>(m));

        //thresholds for stopping the iteration
        const double eps{0.};
        const double eps_p{1.e-6};
        const double eps1{1.e-4};
        const uint_fast64_t maxIter{10};

        //GS: Double-check, from an algorithmic and practical point of view,
        // whether we maintain the two data structures
        // in the main iteration below. :SG
        std::vector<std::vector<T>> W;
        std::set<std::vector<T>> wSet;

        //GS: I believe we will implement a new version
        // of model checker for our purposes. :SG
        mopmc::multiobjective::MOPMCModelChecking<ModelType> scalarisedMOMDPModelChecker(model_);
        //storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<ModelType> scalarisedMOMdpModelChecker(t);

        //Iteration
        uint_fast64_t iter = 0;
        T fDiff = 0;
        while (iter < maxIter && (Phi.size() < 3 || fDiff > eps)) {
            //std::cout << "Iteration: " << iter << "\n";
            std::vector<T> fvt = mopmc::solver::convex::ReLU(vt, h);
            std::vector<T> fvb = mopmc::solver::convex::ReLU(vb, h);
            fDiff = mopmc::solver::convex::diff(fvt, fvb);
            if (!Phi.empty()) {
                // compute the FW and find a new weight vector
                vt = mopmc::solver::convex::frankWolfe(mopmc::solver::convex::reluGradient<T>,
                                                       *vi, 100, W, Phi, h);
                //GS: To be consistent, may change the arg type of
                // reluGradient() to vector. :GS
                Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vt_(vt.data(), vt.size());
                Eigen::Matrix<T, Eigen::Dynamic, 1> cx = h_ - vt_;
                std::vector<T> grad = mopmc::solver::convex::reluGradient(cx);
                //GS: exit if the gradient is very small. :SG
                if (mopmc::solver::convex::l1Norm(grad) < eps_p) { break; }
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
            /*
            std::cout << "W[" << iter << "]: ";
            for (T val: W.back()) {
                std::cout << val << ",";
            }
            std::cout << "\n";
             */
            scalarisedMOMDPModelChecker.check(env_, w);

            uint64_t ini = scalarisedMOMDPModelChecker.getInitialState();
            for (uint_fast64_t i = 0; i < m; ++i) {
                r[i] = scalarisedMOMDPModelChecker.getObjectiveResults()[i][ini];
            }

            Phi.push_back(r);
            LambdaL.push_back(w);
            LambdaR.push_back(r);

            //GS: Compute the initial for frank-wolf and projectedGD.
            // Alright to do it here as the FW function is not called
            // in the first iteration. :SG
            if (Phi.size() == 1) {
                vi = &r;
            } else {
                vi = &vt;
            }

            T wr = std::inner_product(w.begin(), w.end(), r.begin(), static_cast<T>(0.));
            T wvb = std::inner_product(w.begin(), w.end(), vb.begin(), static_cast<T>(0.));
            if (LambdaL.size() == 1 || wr < wvb) {
                T gamma = static_cast<T>(0.1);
                std::cout << "|Phi|: " << Phi.size() << "\n";
                vb = mopmc::solver::convex::projectedGradientDescent(
                        mopmc::solver::convex::reluGradient,
                        *vi, gamma, 10, Phi, W, Phi.size(),
                        h, eps1);
            }

            ++iter;
        }

        //scalarisedMdpModelChecker.multiObjectiveSolver(env_);
        std::cout << "Convex query done! \n";
    }

}
