//
// Created by guoxin on 20/11/23.
//

#ifndef MOPMC_DATA_H
#define MOPMC_DATA_H

#include <string>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/environment/Environment.h>

namespace mopmc {
    template<typename V, typename I>
    struct Data {

        Eigen::SparseMatrix<V> transitionMatrix;
        I rowCount{};
        I colCount{};
        std::vector<I> rowGroupIndices;
        std::vector<I> row2RowGroupMapping;
        std::vector<std::vector<V>> rewardVectors;
        std::vector<V> flattenRewardVector;
        I objectiveCount{};
        std::vector<bool> probObjectives;
        std::vector<V> thresholds;
        std::vector<V> weightedVector;
        std::vector<I> defaultScheduler;
        I initialRow{};

        void bar(const V &t);
    };

}


#endif //MOPMC_DATA_H
