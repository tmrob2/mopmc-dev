//
// Created by guoxin on 27/11/23.
//

#include "TestingQuery.h"

namespace mopmc::queries {

    template<typename T>
    void TestingQuery<T>::query() {
        std::vector<T> w = {-.5, -.5};
        mopmc::Data<double, int> data32 = this->data_.castToGpuData();
        mopmc::value_iteration::gpu::CudaValueIterationHandler<double> cudaVIHandler(
                data32.transitionMatrix,
                data32.rowGroupIndices,
                data32.row2RowGroupMapping,
                data32.flattenRewardVector,
                data32.defaultScheduler,
                data32.initialRow,
                data32.objectiveCount
        );
        cudaVIHandler.initialise();
        cudaVIHandler.valueIterationPhaseOne(w);
        cudaVIHandler.valueIterationPhaseTwo();
        cudaVIHandler.exit();
        std::cout << "----------------------------------------------\n";
        std::cout << "@_@ CUDA VI TESTING OUTPUT: \n";
        std::cout << "weight: [" << w[0] << ", " << w[1] << "]\n";
        std::cout << "Result at initial state ";
        for (int i = 0; i < data32.objectiveCount; ++i) {
            std::cout << "- Objective " << i << ": " << cudaVIHandler.results_[i] << " ";
        }
        std::cout << "\n";
        std::cout << "(Negative) Weighted result: " << cudaVIHandler.results_[data32.objectiveCount] << "\n";
        std::cout << "----------------------------------------------\n";
    }

    template
    class TestingQuery<double>;
}