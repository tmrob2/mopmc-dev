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
    // V value type, I index type
    template<typename V, typename I>
    struct Data {

        Eigen::SparseMatrix<V> transitionMatrix;
        std::vector<std::vector<V>> rewardVectors;
        std::vector<V> flattenRewardVector;
        std::vector<V> thresholds;
        std::vector<V> weightedVector;

        I rowCount{};
        I colCount{};
        I objectiveCount{};
        I initialRow{};
        std::vector<I> rowGroupIndices;
        std::vector<I> row2RowGroupMapping;
        std::vector<I> defaultScheduler;

        std::vector<bool> probObjectives;
        std::vector<bool> optDirections; // Minimize = 0, Maximize = 1

    };

    template struct Data<double, uint64_t>;
}


#endif //MOPMC_DATA_H
