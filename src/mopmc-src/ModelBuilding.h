//
// Created by guoxin on 17/11/23.
//

#ifndef MOPMC_MODELBUILDING_H
#define MOPMC_MODELBUILDING_H

#include <string>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/environment/Environment.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>


namespace  mopmc {

    template<typename M>
    class ModelBuilder {
    public:

        static typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType build(
                std::string const &path_to_model, std::string const &property_string, storm::Environment &env);

        Eigen::SparseMatrix<double> transitionMatrix;
        uint64_t rowCount{};
        uint64_t colCount{};
        std::vector<uint64_t> rowGroupIndices;
        std::vector<uint64_t> row2RowGroupMapping;
        std::vector<std::vector<typename M::ValueType>> rewardVectors;
        std::vector<typename M::ValueType> flattenRewardVector;
        uint64_t objectiveCount{};
        std::vector<bool> probObjectives;
        std::vector<typename M::ValueType> thresholds;
        std::vector<typename M::ValueType> weightedVector;
        std::vector<uint64_t> defaultScheduler;
        uint64_t initialRow{};

    };


}

#endif //MOPMC_MODELBUILDING_H
