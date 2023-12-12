//
// Created by thomas on 22/10/23.
//
#include "Looper.h"
#include "mopmc-src/QueryData.h"
#include "mopmc-src/hybrid-computing/Problem.h"
#include "mopmc-src/solvers/CudaValueIteration.cuh"
#include <iostream>
#include <memory>
#include <storm/adapters/EigenAdapter.h>

namespace hybrid {

    template<typename V>
    CLooper<V>::CLooper(
            uint id_, hybrid::ThreadSpecialisation spec, QueryData<V, int>& model)
            : id(id_), mRunning(false), mAbortRequested(false), mRunnables(), mRunnablesMutex(),
              mDispatcher(std::shared_ptr<CDispatcher>(new CDispatcher(*this))),
              mBusy(false), expectedSolutions(0), threadType(spec) {
        if(spec==ThreadSpecialisation::CPU){
            data = std::make_shared<QueryData<V, int>>(model);
        } else {
            // send the data to the GPU
            sendDataGPU(model);
        }
    }

    // Running the infinite loop
    template <typename V>
    bool CLooper<V>::run() {

        try{
            mThread = std::thread(&CLooper::runFunc, this);
        } catch (...) {
            return false;
        }

        return true;
    }

    template <typename V>
    bool CLooper<V>::stop() {
        abortAndJoin();
        return true;
    }

    template <typename V>
    bool CLooper<V>::getAbortRequested() const {
        return mAbortRequested.load();
    }

    template <typename V>
    bool CLooper<V>::running() const {
        return mRunning.load();
    }

    template <typename V>
    bool CLooper<V>::busy() const {
        return mBusy.load();
    }

    template <typename V>
    boost::optional<typename CLooper<V>::Sch> CLooper<V>::next() {
        // A mutex is required to guard against simultaneous access to the task collection
        // by the worker and dispatching threads.
        std::lock_guard guard(mRunnablesMutex); // only works with Cxx17 but we force this standard in the CMAKE

        boost::optional<typename CLooper<V>::Sch> problem;
        if(!mRunnables.empty()) {
            problem = mRunnables.front();
            mRunnables.pop();
        }

        return problem;

    }

    // explicit instantiations of send data overloaded
    template <typename V>
    void CLooper<V>::sendDataGPU(mopmc::QueryData<V, int>& data) {
        // allocate a matrix
        //std::vector<int> scheduler(data.defaultScheduler.begin(), data.defaultScheduler.end());
        //GS: use the new CudaVIHanlder constructor :SG
        mopmc::value_iteration::gpu::CudaValueIterationHandler<V> cudaVIHandler(&data);
        /*
        mopmc::value_iteration::gpu::CudaValueIterationHandler<V> cudaVIHandler(
                data.transitionMatrix,
                data.rowGroupIndices,
                data.row2RowGroupMapping,
                data.flattenRewardVector,
                data.defaultScheduler,
                data.initialRow,
                data.objectiveCount
        );*/
        cudaVIHandler.initialize();
        gpuData = std::make_shared<mopmc::value_iteration::gpu::CudaValueIterationHandler<V>>(cudaVIHandler);
    }

    //! Empties all of the values in the solution vector
    template <typename V>
    std::vector<int> CLooper<V>::getSolution() {
        std::vector<int> rtn = {};
        if (!solutions.empty()) {
            solutions.swap(rtn);
        }
        return rtn;
    }

    template <typename V>
    void CLooper<V>::runFunc() {
        mRunning.store(true);

        while (!mAbortRequested.load()) {
            //std::cout << "Abort Requested: " << mAbortRequested.load() << "Running: " << mRunning.load() << "\n";
            try {
                // Do something
                using namespace std::chrono_literals;
                //std::cout << "Calling next()\n";
                boost::optional<ThreadProblem<V>> r = next();
                if(r && !r.get().isEmpty()) {
                    std::cout << "r: " << r.get().getFirst() << "\n";
                    mBusy.store(true);
                    r.get()();

                    solutions.push_back(1);
                    mBusy.store(false);
                }
            } catch (std::runtime_error& e) {
                // Something more specific
            } catch (...) {
                // Make sure that nothing leaves the thread for now.
                std::cout << "something else...\n";
            }

        }

        std::cout << "Thread " << id << " stopped\n";

        mRunning.store(false);
    }

    template <typename V>
    void CLooper<V>::abortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        }
    }

    template <typename V>
    bool CLooper<V>::poolAbortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        } else {
            return false;
        }
        return true;
    }

    template <typename V>
    bool CLooper<V>::solutionsReady() {
        if (solutions.size() == expectedSolutions) {
            return true;
        }
        return false;
    }

    template <typename V>
    bool CLooper<V>::post(typename CLooper<V>::Sch &&aRunnable) {
        if(not running()) {
            // deny insertion
            std::cout << "Looper not running\n";
            return false;
        }

        try {
            std::cout << "thread type" << this->threadType << std::endl;
            switch (this->threadType) {
                case ThreadSpecialisation::CPU:
                    aRunnable.setProblemData(data);
                    std::cout << "set problem data cpu\n";
                    break;
                case ThreadSpecialisation::GPU:
                    // allocate GPU data
                    std::cout << "set problem data gpu\n";
                    aRunnable.setProblemData(gpuData);
                    break;
            }
            std::lock_guard guard(mRunnablesMutex);
            mRunnables.push(std::move(aRunnable));
            expectedSolutions += 1;
        } catch (...) {
            return false;
        }

        return true;
    }

    template <typename V>
    void CLooperPool<V>::stop() {
        // for each running thread stop it
        for (const auto & looper: mLoopers) {
            looper->stop();
        }
    }

    template <typename V>
    bool CLooperPool<V>::run() {
        // start each thread in the threadpool
        std::vector<bool> started(mLoopers.size(), false);
        for(uint i = 0; i < mLoopers.size(); ++i) {
            started[i] = mLoopers[i]->run();
        }
        bool allStarted = std::all_of(started.begin(), started.end(), [](bool val) {
            return val;
        });
        if (allStarted) {
            return true;
        } else {
            return false;
        }
    }

    template <typename V>
    bool CLooperPool<V>::running() {
        bool check = true;
        for (const auto & mLooper : mLoopers) {
            if(!mLooper->running()) {
                return false;
            }
        }
        return true;
    }

    template<typename V>
    void CLooperPool<V>::collectSolutions() {
        bool allLoopersReady = false;
        while(!allLoopersReady) {
            allLoopersReady = true;
            for (const auto & mLooper : mLoopers) {
                if(!mLooper->solutionsReady()) {
                    allLoopersReady = false;
                }
            }
        }
        //
        for(const auto & mLooper : mLoopers){
            auto sol = mLooper->getSolution();
            solutions.insert(solutions.end(), sol.begin(), sol.end());
        }
    }

    template <typename V>
    std::vector<std::shared_ptr<typename CLooper<V>::CDispatcher>> CLooperPool<V>::getDispatchers() {
        std::vector<std::shared_ptr<typename CLooper<V>::CDispatcher>> dispatchers;
        for (const auto & mLooper : mLoopers) {
            dispatchers.push_back(mLooper->getDispatcher());
        }
        return dispatchers;
    }

    template <typename V>
    void CLooperPool<V>::solve(std::vector<hybrid::ThreadProblem<V>> tasks) {
        // implementation of the scheduler solver

        auto dispatchers = getDispatchers();
        uint countDown = tasks.size();
        uint k = 0;
        while(countDown > 0) {

            // Allocation is about thread specialisations
            //k = k % mLoopers.size();
            hybrid::ThreadProblem task = tasks.back();

            if (task.getSpec() == ThreadSpecialisation::CPU){
                // allocate the data to the CPU thread
                if (dispatchers[0]->post(std::move(task))) {
                    std::cout << "CPU Thread accepted the task\n";
                    --countDown;
                    tasks.pop_back();
                }
            } else if (task.getSpec() == ThreadSpecialisation::GPU) {
                if (dispatchers[1]->post(std::move(task))) {
                    std::cout  << "GPU Thread accepted the task\n"; 
                    --countDown;
                    tasks.pop_back();
                }
            }

            
        }

        collectSolutions();
    }

    template<typename V>
    std::vector<int>& CLooperPool<V>::getSolutions() {
        return solutions;
    }   

    template class CLooperPool<double>;
    template class CLooper<double>;
}