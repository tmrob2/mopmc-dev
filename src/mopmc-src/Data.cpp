//
// Created by guoxin on 20/11/23.
//

#include "Data.h"

namespace mopmc {


    template<typename V, typename I>
    Data<double, int> Data<V,I>::castToGpuData() {
        assert(typeid(V)== typeid(double));
        assert(this->transitionMatrix.nonZeros() < INT_MAX);
        assert(this->rowCount < INT_MAX);

        Data<double, int>  data4Gpu;
        //non cast
        data4Gpu.transitionMatrix = this->transitionMatrix;
        data4Gpu.rewardVectors = this->rewardVectors;
        data4Gpu.flattenRewardVector = this->flattenRewardVector;
        data4Gpu.thresholds = this->thresholds;
        data4Gpu.weightedVector = this->weightedVector;

        data4Gpu.probObjectives = this->probObjectives;

        //cast
        data4Gpu.rowCount = (int) this->rowCount;
        data4Gpu.colCount = (int) this->colCount;
        data4Gpu.objectiveCount = (int) this->objectiveCount;
        data4Gpu.initialRow = (int) this->initialRow;

        std::vector<int> rowGroupIndices1(this->rowGroupIndices.begin(), this->rowGroupIndices.end());
        data4Gpu.rowGroupIndices = std::move(rowGroupIndices1);

        std::vector<int> rowToRowGroupMapping1(this->row2RowGroupMapping.begin(), this->row2RowGroupMapping.end());
        data4Gpu.row2RowGroupMapping = std::move(rowToRowGroupMapping1);

        std::vector<int> scheduler1(this->defaultScheduler.begin(), this->defaultScheduler.end());
        data4Gpu.defaultScheduler = std::move(scheduler1);

        return data4Gpu;
    }

    template
    struct Data<double, uint64_t>;

}
