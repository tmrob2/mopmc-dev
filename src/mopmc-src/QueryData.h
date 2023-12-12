//
// Created by guoxin on 20/11/23.
//

#ifndef MOPMC_QUERYDATA_H
#define MOPMC_QUERYDATA_H

#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>

namespace mopmc {
    // V value type, I index type
    template<typename V, typename I>
    struct QueryData {

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

        std::vector<bool> isProbabilisticObjective;
        std::vector<bool> isThresholdUpperBound;

    };

    template struct QueryData<double, uint64_t>;
}


#endif //MOPMC_QUERYDATA_H
