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

#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include "../solvers/ConvexQuery.h"
#include "../solvers/CudaOnlyValueIteration.h"

#include <storm/transformer/GoalStateMerger.h>
#include <storm/transformer/EndComponentEliminator.h>

namespace mopmc::queries {

    typedef typename ModelType::ValueType T;

    ConvexQuery::ConvexQuery(const PrepReturnType &model) : model_(model) {}

    ConvexQuery::ConvexQuery(const PrepReturnType &model,
                             const storm::Environment &env) :
            model_(model), env_(env) {}

    void ConvexQuery::query() {

        //Data generation
        const uint64_t m = model_.objectives.size(); // m: number of objectives
        //const uint64_t n = model_.preprocessedModel->getNumberOfChoices(); // n: number of state-action pairs
        //const uint64_t k = model_.preprocessedModel->getNumberOfStates(); // k: number of states

        // There seems to be something extra going on with the EC quotient. Tried performing VI
        // with preprocessed amd preprocessed + initialised but neither seems to converge.
        FindBugInModelExtraProcessing<ModelType> extraProcModel;
        extraProcModel.initialise(model_, model_.objectives);

        std::vector<std::vector<T>> rho(m);
        std::vector<T> rho_flat(extraProcModel.transitionMatrix.getRowCount()*m);//rho: all reward vectors
        //GS: need to store whether an objective is probabilistic or reward-based.
        //TODO In future we will use treat them differently in the loss function. :GS
        /*std::vector<bool> isProbObj(m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            auto &name_ = model_.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
            rho[i] = model_.preprocessedModel->getRewardModel(name_)
                    .getTotalRewardVector(model_.preprocessedModel->getTransitionMatrix());
            for (uint_fast64_t j = 0; j < n; ++j) {
                rho_flat[i * n + j] = rho[i][j];
            }
            isProbObj[i] = model_.objectives[i].originalFormula->isProbabilityOperatorFormula();
        }*/

        std::vector<bool> isProbObj(m);
        for (uint_fast64_t i = 0; i < m; ++i) {
            auto &name = model_.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
            for (uint_fast64_t j = 0; j < extraProcModel.transitionMatrix.getRowCount(); ++j) {
                rho_flat[i * extraProcModel.transitionMatrix.getRowCount() + j] = extraProcModel.actionRewards[i][j];
            }
        }

        // See if we have to make the initial EC Quotient after this point
        std::vector<T> w { static_cast<T>(1.0), static_cast<T>(1.0) };
        std::vector<T> wR(extraProcModel.transitionMatrix.getRowCount());
        weightedRewards(rho_flat, w, wR);
        extraProcModel.update(wR);

        const uint64_t n = extraProcModel.ecqTransitionMatrix.getRowCount();
        const uint64_t k = extraProcModel.ecqTransitionMatrix.getRowGroupCount();

        //auto P = // P: transition matrix as eigen sparse matrix
        //        storm::adapters::EigenAdapter::toEigenSparseMatrix(model_.preprocessedModel->getTransitionMatrix());
        //auto P = storm::adapters::EigenAdapter::toEigenSparseMatrix(extraProcModel.transitionMatrix);
        auto P = storm::adapters::EigenAdapter::toEigenSparseMatrix(extraProcModel.ecqTransitionMatrix);
        P->makeCompressed();

        std::cout << "P init (" << P->rows() << ", " << P->cols() << ")\n";
        std::cout << "PreModel (" << model_.preprocessedModel->getTransitionMatrix().getRowCount() << ", " << model_.preprocessedModel->getTransitionMatrix().getColumnCount() << ")\n";
        std::cout << "sizes k " << k << " n " << n << " m " << m << "\n";

        std::vector<uint64_t> pi(k, static_cast<uint64_t>(0)); // pi: scheduler
        //std::vector<uint64_t> stateIndices = model_.preprocessedModel->getTransitionMatrix().getRowGroupIndices();
        std::vector<uint64_t> stateIndices = extraProcModel.ecqTransitionMatrix.getRowGroupIndices();

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
        //std::vector<T> w = // w: weight vector
        w = std::vector<T>(m, static_cast<T>(1.0) / static_cast<T>(m));
        std::vector<T> x(k, static_cast<T>(0.)); //x: state values
        std::vector<T> y(n, static_cast<T>(0.)); //y: state-action values

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

        //mopmc::value_iteration::cuda_only::CudaIVHandler<ModelType::ValueType> cudaIvHandler(*P,rho_flat);
        mopmc::value_iteration::cuda_only::CudaIVHandler<ModelType::ValueType>
                cudaIvHandler(*P, stateIndices,
                              extraProcModel.ecqToOriginalChoiceMapping, // extra argument for mapping rewards from orig to ecQuotient states
                              extraProcModel.transitionMatrix.getRowCount(),// extra size argument for downscaling
                              rho_flat,pi, w, x,y);
        cudaIvHandler.initialise();
        //cudaIvHandler.valueIteration();
        cudaIvHandler.valueIterationPhaseOne(w);
        cudaIvHandler.exit();

        return;

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

    // Can delete this later, just used for EcQuotient analysis
    void ConvexQuery::weightedRewards(std::vector<T>& flatRewards,
                                      std::vector<T>& w,
                                      std::vector<T>& rW){
        // assumes rW initialised at zero
        uint n = rW.size();
        uint nobj = w.size();
        std::cout << "|rW| " << n << "\n";
        std::cout << "|flat rewards|: " << flatRewards.size() << "\n";
        for (uint_fast64_t i = 0; i < n; ++i) {
            for(uint_fast64_t j = 0; j < nobj; ++j) {
                rW[i] += w[j] * flatRewards[j * n + i];
            }
        }
    }

    template <typename SparseModelType>
    void FindBugInModelExtraProcessing<SparseModelType>::initialise(mopmc::PrepReturnType const& preprocessorResult,
                                                                    std::vector<storm::modelchecker::multiobjective::Objective<typename SparseModelType::ValueType>>& objectives) {

        auto rewardAnalysis = storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectiveRewardAnalysis<SparseModelType>::analyze(preprocessorResult);

        storm::storage::BitVector maybeStates = rewardAnalysis.totalRewardLessInfinityEStates.get() & ~rewardAnalysis.reward0AStates;
        storm::storage::BitVector finiteTotalRewardChoices = preprocessorResult.preprocessedModel->getTransitionMatrix().getRowFilter(
                rewardAnalysis.totalRewardLessInfinityEStates.get(), rewardAnalysis.totalRewardLessInfinityEStates.get());
        std::set<std::string> relevantRewardModels;
        for (auto const& obj : objectives) {
            obj.formula->gatherReferencedRewardModels(relevantRewardModels);
        }

        storm::transformer::GoalStateMerger<SparseModelType> merger(*preprocessorResult.preprocessedModel);
        auto mergerResult = merger.mergeTargetAndSinkStates(
                maybeStates,
                rewardAnalysis.reward0AStates, storm::storage::BitVector(maybeStates.size(), false),
                std::vector<std::string>(relevantRewardModels.begin(), relevantRewardModels.end()),
                finiteTotalRewardChoices);

        initialiseModelSpecficData(*mergerResult.model, objectives);

        transitionMatrix = std::move(mergerResult.model->getTransitionMatrix());
        initialState = *mergerResult.model->getInitialStates().begin();
        totalReward0EStates = rewardAnalysis.totalReward0EStates % maybeStates;
        if (mergerResult.targetState) {
            std::cout << "merger result target state is true\n";
            // There is an additional state in the result
            totalReward0EStates.resize(totalReward0EStates.size() + 1, true);

            // The overapproximation for the possible ec choices consists of the states that can reach the target states with prob. 0 and the target state itself.
            storm::storage::BitVector targetStateAsVector(transitionMatrix.getRowGroupCount(), false);
            targetStateAsVector.set(*mergerResult.targetState, true);
            ecChoicesHint = transitionMatrix.getRowFilter(
                    storm::utility::graph::performProb0E(transitionMatrix, transitionMatrix.getRowGroupIndices(), transitionMatrix.transpose(true),
                                                         storm::storage::BitVector(targetStateAsVector.size(), true), targetStateAsVector));
            ecChoicesHint.set(transitionMatrix.getRowGroupIndices()[*mergerResult.targetState], true);
        } else {
            std::cout << "merger result target state is false\n";
            ecChoicesHint = storm::storage::BitVector(transitionMatrix.getRowCount(), true);
        }
    }

    template <typename SparseModelType>
    void FindBugInModelExtraProcessing<SparseModelType>::update(std::vector<typename SparseModelType::ValueType> const& weightedRewardVector){
        storm::storage::BitVector newTotalReward0Choices = storm::utility::vector::filterZero(weightedRewardVector);
        if(!setFlag || origReward0Choices != newTotalReward0Choices) {
            auto nonZeroRewardStates = transitionMatrix.getRowGroupFilter(newTotalReward0Choices, true);
            nonZeroRewardStates.complement();
            storm::storage::BitVector subsystemStates = storm::utility::graph::performProbGreater0E(
                    transitionMatrix.transpose(true),
                    storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true), nonZeroRewardStates);


            // Remove neutral end components, i.e., ECs in which no total reward is earned.
            // Note that such ECs contain one (or maybe more) LRA ECs.
            auto ecElimResult = storm::transformer::EndComponentEliminator<typename SparseModelType::ValueType>::transform(
                    transitionMatrix, subsystemStates, ecChoicesHint & newTotalReward0Choices, totalReward0EStates);

            storm::storage::BitVector rowsWithSumLessOne_(ecElimResult.matrix.getRowCount(), false);
            for (uint64_t row = 0; row < rowsWithSumLessOne.size(); ++row) {
                if (ecElimResult.matrix.getRow(row).getNumberOfEntries() == 0) {
                    rowsWithSumLessOne_.set(row, true);
                } else {
                    for (auto const& entry : transitionMatrix.getRow(ecElimResult.newToOldRowMapping[row])) {
                        if (!subsystemStates.get(entry.getColumn())) {
                            rowsWithSumLessOne_.set(row, true);
                            break;
                        }
                    }
                }
            }

            ecqTransitionMatrix = std::move(ecElimResult.matrix);
            ecqToOriginalChoiceMapping = std::move(ecElimResult.newToOldRowMapping);
            originalToEcqStateMapping = std::move(ecElimResult.oldToNewStateMapping);
            ecqToOriginalstateMapping.resize(ecqTransitionMatrix.getRowGroupCount());
            for (uint64_t state = 0; state < originalToEcqStateMapping.size(); ++state) {
                uint64_t ecqState = originalToEcqStateMapping[state];
                if(ecqState < ecqTransitionMatrix.getRowGroupCount()) {
                    ecqToOriginalstateMapping[ecqState].insert(state);
                }
            }
            ecqStateInEcChoices = std::move(ecElimResult.sinkRows);
            origReward0Choices = newTotalReward0Choices;
            origTotalReward0Choices = std::move(newTotalReward0Choices);
            rowsWithSumLessOne = std::move(rowsWithSumLessOne_);

            std::cout << "Weighted rewards size: " << weightedRewardVector.size() << "\n";
            //std::cout << "Ec rewards size: " << ecRewards.size() << "\n";
            ecRewards.resize(ecqTransitionMatrix.getRowCount());


            storm::utility::vector::selectVectorValues(ecRewards, ecqToOriginalChoiceMapping, weightedRewardVector);
        }
    }

    template<typename SparseModelType>
    void FindBugInModelExtraProcessing<SparseModelType>::initialiseModelSpecficData(
            SparseModelType &model,
            std::vector<storm::modelchecker::multiobjective::Objective<typename SparseModelType::ValueType>>& objectives) {
        // set the state action rewards. Also do some sanity checks on the objectives
        this->actionRewards.resize(objectives.size());
        std::cout << "MDP model setup -> Objectives: " <<objectives.size() << "\n";
        for(uint_fast64_t objIndex = 0; objIndex < objectives.size(); ++objIndex){
            auto const& formula = *objectives[objIndex].formula;
            if (!(formula.isRewardOperatorFormula() && formula.asRewardOperatorFormula().hasRewardModelName())){
                std::stringstream ss;
                ss << "Unexpected type of operator formula: " << formula;
                throw std::runtime_error(ss.str());
            }
            if (formula.getSubformula().isCumulativeRewardFormula()) {
                auto const& cumulativeRewardFormula = formula.getSubformula().asCumulativeRewardFormula();
                if (!(!cumulativeRewardFormula.isMultiDimensional() && !cumulativeRewardFormula.getTimeBoundReference().isRewardBound())){
                    std::stringstream ss;
                    ss << "Unexpected type of sub-formula: " << formula.getSubformula();
                    throw std::runtime_error(ss.str());
                }
            } else {
                if (!(formula.getSubformula().isTotalRewardFormula() || formula.getSubformula().isLongRunAverageRewardFormula())){
                    std::stringstream ss;
                    ss << "Unexpected type of sub-formula: " << formula.getSubformula();
                    throw std::runtime_error(ss.str());
                }
            }
            typename SparseModelType::RewardModelType const& rewModel = model.getRewardModel(formula.asRewardOperatorFormula().getRewardModelName());

            if (rewModel.hasTransitionRewards()) {
                throw std::runtime_error("Reward model has transition rewards which is not expected.");
            }
            actionRewards[objIndex] = rewModel.getTotalRewardVector(model.getTransitionMatrix());
        }
    }

    template class FindBugInModelExtraProcessing<storm::models::sparse::Mdp<double>>;

}
