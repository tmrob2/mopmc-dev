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
#include "../solvers/CudaOnlyValueIteration.h"
#include "../solvers/CudaValueIteration.cuh"


namespace mopmc::queries {

    // typedef
    typedef storm::models::sparse::Mdp<double> ModelType;
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType PrepReturnType;
    typedef Eigen::SparseMatrix<typename ModelType::ValueType, Eigen::RowMajor> EigenSpMatrix;
    typedef typename ModelType::ValueType T;

    ConvexQuery::ConvexQuery(const PreprocessedData<ModelType> &data, const storm::Environment &env)
    : data_(data), env_(env) {

    }

    void ConvexQuery::query() {

        //CUDA ONLY TEST BLOCK
        {
            std::vector<T> w = {-.5, -.5};
            assert(data_.objectiveCount == 2);
            assert(data_.transitionMatrix.nonZeros() < INT_MAX);
            assert(data_.rowCount < INT_MAX);
            //Convert data from uint_64 to int
            std::vector<int> rowGroupIndices1(data_.rowGroupIndices.begin(), data_.rowGroupIndices.end());
            std::vector<int> rowToRowGroupMapping1(data_.row2RowGroupMapping.begin(), data_.row2RowGroupMapping.end());
            std::vector<int> scheduler1(data_.defaultScheduler.begin(), data_.defaultScheduler.end());

            mopmc::value_iteration::gpu::CudaValueIterationHandler<double> cudaVIHandler(
                    data_.transitionMatrix,
                    rowGroupIndices1,
                    rowToRowGroupMapping1,
                    data_.flattenRewardVector,
                    scheduler1, w,
                    (int) data_.initialRow
                    );
            cudaVIHandler.initialise();
            cudaVIHandler.valueIterationPhaseOne(w);
            cudaVIHandler.valueIterationPhaseTwo();
            cudaVIHandler.exit();
            {
                std::cout << "----------------------------------------------\n";
                std::cout << "@_@ CUDA VI TESTING OUTPUT: \n";
                std::cout << "weight: [" << w[0] << ", " << w[1] << "]\n";
                std::cout << "Result at initial state ";
                for (int i = 0; i < data_.objectiveCount; ++i) {
                    std::cout << "- Objective " << i << ": " << cudaVIHandler.results_[i] << " ";
                }std::cout <<"\n";
                std::cout << "(Negative) Weighted result: " << cudaVIHandler.results_[2] << "\n";
                std::cout << "----------------------------------------------\n";
            }
        }


        //Data generation
        const uint64_t m = data_.objectiveCount; // m: number of objectives
        const uint64_t n = data_.rowCount; // n: number of choices / state-action pairs
        const uint64_t k = data_.colCount; // k: number of states
        Eigen::SparseMatrix<T> *P = &data_.transitionMatrix;
        assert(data_.rowGroupIndices.size()==k+1);

        std::vector<std::vector<T>> rho(m);
        std::vector<T> rho_flat(n * m);//rho: all reward vectors
        //GS: need to store whether an objective is probabilistic or reward-based.
        //TODO In future we will use treat them differently in the loss function. :GS
        //Initialisation
        std::vector<std::vector<T>> Phi;
        // LambdaL, LambdaR represent Lambda
        std::vector<std::vector<T>> LambdaL;
        std::vector<std::vector<T>> LambdaR;
        std::vector<std::vector<T>> W;
        std::set<std::vector<T>> wSet;


        std::vector<T> h = data_.thresholds;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> h_(h.data(), h.size());

        //vt, vb
        std::vector<T> vt = std::vector<T>(m, static_cast<T>(0.0));
        std::vector<T> vb = std::vector<T>(m, static_cast<T>(0.0));
        //vi: initial vector for Frank-Wolfe
        std::vector<T> *vi;
        std::vector<T> r(m);
        std::vector<T> w = // w: weight vector
                std::vector<T>(m, static_cast<T>(-1.0) / static_cast<T>(m));
        //thresholds for stopping the iteration
        const double eps{0.};
        const double eps_p{1.e-6};
        const double eps1{1.e-4};
        const uint_fast64_t maxIter{20};

        //GS: I believe we will implement a new version
        // of model checker for our purposes. :SG
       //mopmc::multiobjective::MOPMCModelChecking<ModelType> scalarisedMOMDPModelChecker(model_);

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

            //scalarisedMOMDPModelChecker.check(env_, w);
            /*
            uint64_t ini = data_.initialRow getInitialState();
            for (uint_fast64_t i = 0; i < m; ++i) {
                r[i] = scalarisedMOMDPModelChecker.getObjectiveResults()[i][ini];
            }
             */


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
