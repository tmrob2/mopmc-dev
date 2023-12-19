//
// Created by thomas on 05/11/23.
//

#ifndef MOPMC_PROBLEM_H
#define MOPMC_PROBLEM_H

#include <cstdint>
#include <memory>
#include <storm/storage/SparseMatrix.h>
#include <storm/utility/constants.h>
#include <Eigen/Sparse>
#include <vector>
#include "mopmc-src/Data.h"
#include "mopmc-src/solvers/CudaValueIteration.cuh"

namespace hybrid {

enum ThreadSpecialisation {
    GPU,
    CPU
};

enum Problem {
    Scheduler,
    DTMC
};

template<typename ValueType>
class ThreadProblem {
public:
    //! The problem needs to solve a value iteration problem.
    //! Take the inputs for the value iteration problem
    ThreadProblem(uint index_, std::vector<ValueType> w_, ThreadSpecialisation spec_, Problem probType_):
    w(w_), spec(spec_), probType(probType_), index(index_), empty(false) {
        // Intentionally left blank
    }

    ThreadProblem(uint index_, ThreadSpecialisation spec_, Problem probType_, int start_obj, int end_obj) :
    spec(spec_), probType(probType_), empty(false), startObj(start_obj), endObj(end_obj) {
        // Intentionally left blank
    }

    void setEmpty() {
        this->empty = true;
    }

    bool &isEmpty() {
        return this->empty;
    }

    uint getFirst() const {
        return index;
    }

    void setProblemData(std::shared_ptr<mopmc::Data<double, int>> cpuData){
        data = cpuData;
    }

    void setProblemData(std::shared_ptr<mopmc::value_iteration::gpu::CudaValueIterationHandler<double>>& gpuData_){
        gpuData = gpuData_;
    }

    ThreadSpecialisation getSpec() { return spec; }

    // SchedulerProblem callable
    // The goal of a scheduler problem is to compute an optimal scheduler. 
    int operator()() const {
        //std::cout << "function called..\n";
        std::cout << "spec: " << spec << "\n";
        switch (probType) {
            case Problem::Scheduler:
                if (spec == ThreadSpecialisation::CPU) {
                    // do VI ops on CPU
                } else {
                    // do VI ops on GPU
                    std::cout << "starting VI ph 1\n";
                    gpuData->valueIterationPhaseOne(w, true);
                }
                break;
            case Problem::DTMC:
                // do some DTMC stuff
                if (spec == ThreadSpecialisation::CPU) {
                    // do DTMC ops on the CPU
                } else {
                    // do DTMC ops on the GPU
                    std::cout << "starting VI ph 2\n";
                    gpuData->valueIterationPhaseTwo_v2(this->startObj, this->endObj);
                }
                break;
        }
        return 1;
        // A scheduler problem is relatively easy to solve
    }

private:
    bool empty;
    uint index;
    ThreadSpecialisation spec;
    std::shared_ptr<mopmc::Data<double, int>> data;
    std::shared_ptr<mopmc::value_iteration::gpu::CudaValueIterationHandler<double>> gpuData;
    std::vector<ValueType> w;
    std::shared_ptr<std::vector<uint64_t>> scheduler;
    Problem probType;
    int startObj;
    int endObj;
    std::vector<double> ObjResults;
};
}
#endif //MOPMC_PROBLEM_H
