//
// Created by guoxin on 27/11/23.
//

#include "TestingQuery.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void TestingQuery<T, I>::query() {
        std::vector<T> w = {-.5, -.5};
        mopmc::value_iteration::gpu::CudaValueIterationHandler<double> cudaVIHandler(
                this->data_.transitionMatrix,
                this->data_.rowGroupIndices,
                this->data_.row2RowGroupMapping,
                this->data_.flattenRewardVector,
                this->data_.defaultScheduler,
                this->data_.initialRow,
                this->data_.objectiveCount
        );
        cudaVIHandler.initialise();
        cudaVIHandler.valueIterationPhaseOne(w);
        cudaVIHandler.valueIterationPhaseTwo();
        cudaVIHandler.exit();
        std::cout << "----------------------------------------------\n";
        std::cout << "@_@ CUDA VI TESTING OUTPUT: \n";
        std::cout << "weight: [" << w[0] << ", " << w[1] << "]\n";
        std::cout << "Result at initial state ";
        for (int i = 0; i < this->data_.objectiveCount; ++i) {
            std::cout << "- Objective " << i << ": " << cudaVIHandler.results_[i] << " ";
        }
        std::cout << "\n";
        std::cout << "(Negative) Weighted result: " << cudaVIHandler.results_[this->data_.objectiveCount] << "\n";
        std::cout << "----------------------------------------------\n";
    }

    template
    class TestingQuery<double, int>;
}