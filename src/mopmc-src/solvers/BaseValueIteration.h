//
// Created by guoxin on 12/12/23.
//

#ifndef MOPMC_BASEVALUEITERATION_H
#define MOPMC_BASEVALUEITERATION_H

#include <vector>
#include <Eigen/Sparse>
#include "../Data.h"

namespace mopmc::value_iteration {

    template<typename V>
    class BaseValueIteration {
    public:
        /*
        explicit BaseValueIteration(mopmc::Data<V, int> data) {
            transitionMatrix = data.transitionMatrix;
            flattenRewardVector = data.flattenRewardVector;
            scheduler = data.defaultScheduler;
            rowGroupIndices = data.rowGroupIndices;
            iniRow = data.initialRow;
            nobjs = data.objectiveCount;
        }*/

        virtual int initialize() {return  0;};
        virtual int exit() {return 0;}

        virtual int valueIteration(const std::vector<double> &w) {return 0;}

        Eigen::SparseMatrix<V, Eigen::RowMajor> transitionMatrix;
        std::vector<V> flattenRewardVector;
        std::vector<int> scheduler;
        std::vector<int> rowGroupIndices;
        std::vector<int> row2RowGroupMapping;
        std::vector<double> weightedValueVector;
        std::vector<double> results;
        double weightedResult{};
        int iniRow{};
        int nobjs{};

    };
}

#endif //MOPMC_BASEVALUEITERATION_H
