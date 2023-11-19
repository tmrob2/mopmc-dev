//
// Created by guoxin on 17/11/23.
//

#ifndef MOPMC_PREPROCESSING_H
#define MOPMC_PREPROCESSING_H

#include <string>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/environment/Environment.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>


namespace  mopmc {

    template<typename M>
    class PreprocessedData {
    public:
        PreprocessedData();

        explicit PreprocessedData(typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn);

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

    template<typename M>
    //void preprocess(std::string const& path_to_model, std::string const& property_string, storm::Environment env);
    PreprocessedData<M> preprocess(
            std::string const &path_to_model,
            std::string const &property_string,
            storm::Environment &env);


}

#endif //MOPMC_PREPROCESSING_H
