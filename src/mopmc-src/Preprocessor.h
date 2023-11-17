//
// Created by guoxin on 17/11/23.
//

#ifndef MOPMC_PREPROCESSOR_H
#define MOPMC_PREPROCESSOR_H

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

        PreprocessedData(typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn);

        Eigen::SparseMatrix<double> transitionMatrix;
        uint64_t rowCount;
        uint64_t colCount;
        std::vector<uint64_t> rowGroupIndices;
        std::vector<uint64_t> row2RowGroupMapping;
        std::vector<std::vector<typename M::ValueType>> rewardVectors;
        std::vector<typename M::ValueType> flattenRewardVector;
        uint64_t numObjs;
        std::vector<bool> isProbObj;
        std::vector<typename M::ValueType> thresholds;
        std::vector<typename M::ValueType> weightedVector;
        std::vector<uint64_t> defaultScheduler;
        uint64_t iniRow;

    };

    template<typename M>
    //void preprocess(std::string const& path_to_model, std::string const& property_string, storm::Environment env);
    PreprocessedData<typename M::ValueType> preprocess(std::string const& path_to_model, std::string const& property_string, storm::Environment env);

}

#endif //MOPMC_PREPROCESSOR_H
