//
// Created by guoxin on 2/11/23.
//
#ifndef MOPMC_CONVEXXQUERY_H
#define MOPMC_CONVEXXQUERY_H
#include "../Runner.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/api/storm.h>

namespace mopmc::queries {

    class ConvexQuery;

    // TODO: Investigating whether we need to do this part or even some of this part, and therefore
    //  which parts are necessary to call our convex query and allow convergence.
    //
    // This class does extra preprocessing of model
    // 1. It initialises the model with mergerModel: Why is this not enough to converge?
    // 2. It creates the ecQuotient which collapses all EC loops to a single state
    //    and possibly adds some extra sink state?
    template <typename SparseModelType>
    class FindBugInModelExtraProcessing {
    public:
        typedef typename SparseModelType::ValueType T;
        FindBugInModelExtraProcessing() : setFlag(false) {}
        void update(std::vector<T> const& weightedRewardVector);

        void initialise(PrepReturnType const& preprocessorResult,
                        std::vector<storm::modelchecker::multiobjective::Objective<T>>& objectives);

        void initialiseModelSpecficData(SparseModelType& model,
                                        std::vector<storm::modelchecker::multiobjective::Objective<T>>& objectives);

        friend class ConvexQuery;
    private:
        storm::storage::SparseMatrix<T> ecqTransitionMatrix;
        std::vector<uint_fast64_t> ecqToOriginalChoiceMapping;
        std::vector<uint_fast64_t> originalToEcqStateMapping;
        std::vector<storm::storage::FlatSetStateContainer> ecqToOriginalstateMapping;
        storm::storage::BitVector ecqStateInEcChoices;
        storm::storage::BitVector origReward0Choices;
        storm::storage::BitVector origTotalReward0Choices;
        storm::storage::BitVector rowsWithSumLessOne;

        // Seems to be another layer of preprocessing missed through model_.initialise() which
        // is not called

        storm::storage::SparseMatrix<T> transitionMatrix;
        storm::storage::BitVector ecChoicesHint;
        uint64_t initialState;
        storm::storage::BitVector totalReward0EStates;
        bool setFlag;
        std::vector<std::vector<T>> actionRewards;

        //
        std::vector<T> ecRewards;
    };


    class ConvexQuery{
        typedef typename ModelType::ValueType T;

    public:
        explicit ConvexQuery(const PrepReturnType& model);
        ConvexQuery(const PrepReturnType& model, const storm::Environment& env);

        void query();

        // Can delete this later, just used for analysis of what EcQ is doing
        void weightedRewards(std::vector<T>& flatRewards, std::vector<T>& w, std::vector<T>& rW);

        PrepReturnType model_;
        storm::Environment env_;
    };
    //TODO



}

#endif //MOPMC_CONVEXXQUERY_H
