//
// Created by thomas on 05/11/23.
//

#ifndef MOPMC_HYBRIDCONTROLLER_H
#define MOPMC_HYBRIDCONTROLLER_H

#include "Looper.h"
#include "mopmc-src/hybrid-computing/Problem.h"

namespace hybrid {

template <typename ValueType, typename D>
void controller(D const& data) { // probably move some data into this function
    typedef hybrid::ThreadProblem<ValueType> P;
    //std::vector<std::unique_ptr<CLooper<P, ValueType>>> threads(2);

    auto cpuThread = std::make_unique<CLooper<P, ValueType>>(0, ThreadSpecialisation::CPU, data);
    auto gpuThread = std::make_unique<CLooper<P, ValueType>>(1, ThreadSpecialisation::GPU, data);

    // the threads should already start with their assigned data
    // this means that we don't need to assign data to them at a later stage

    auto threadPool = CLooperPool<P, ValueType>(std::move({cpuThread, gpuThread}));

    std::cout << "Starting the thread pool..\n";

    threadPool.run();

    while(!threadPool.running()) {
        // do nothing
    }

    std::cout << "Thread pool started\n";

    // move some data into the thread pool

    threadPool.assignGlobalData(data);

}




}

#endif //MOPMC_HYBRIDCONTROLLER_H
