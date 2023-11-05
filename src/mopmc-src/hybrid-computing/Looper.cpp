//
// Created by thomas on 22/10/23.
//
#include "Looper.h"
#include <iostream>

namespace mythread {

    // Running the infinite loop
    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::run() {

        try{
            mThread = std::thread(&CLooper::runFunc, this);
        } catch (...) {
            return false;
        }

        return true;
    }

    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::stop() {
        abortAndJoin();
        return true;
    }

    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::getAbortRequested() const {
        return mAbortRequested.load();
    }

    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::running() const {
        return mRunning.load();
    }

    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::busy() const {
        return mBusy.load();
    }

    template <typename T, typename ValueType>
    boost::optional<T> CLooper<T, ValueType>::next() {
        // A mutex is required to guard against simultaneous access to the task collection
        // by the worker and dispatching threads.
        std::lock_guard guard(mRunnablesMutex); // only works with Cxx17 but we force this standard in the CMAKE

        boost::optional<T> problem;
        if(!mRunnables.empty()) {
            problem = mRunnables.front();
            mRunnables.pop();
        }

        return problem;

    }

    // explicit instantiations of send data overloaded
    template <typename T, typename ValueType>
    void sendDataGPU(typename CLooper<T, ValueType>::SpMat& matrix,
                     std::vector<int> const& rowGroupIndices,
                     std::vector<int>& pi) {
        // allocate a matrix
        cuTransitionMatrix (matrix, rowGroupIndices, pi);
    }

    template <typename T, typename ValueType>
    void sendDataCPU(typename CLooper<T, ValueType>::SpMat& matrix) {
        // let this thread take ownership of the transition matrix for CPU operations
    }

    //! Empties all of the values in the solution vector
    template <typename T, typename ValueType>
    std::vector<std::pair<int, double>> CLooper<T, ValueType>::getSolution() {
        std::vector<std::pair<int, double>> rtn = {};
        if (!solutions.empty()) {
            solutions.swap(rtn);
        }
        return rtn;
    }

    template <typename T, typename ValueType>
    void CLooper<T, ValueType>::runFunc() {
        mRunning.store(true);

        while (!mAbortRequested.load()) {
            //std::cout << "Abort Requested: " << mAbortRequested.load() << "Running: " << mRunning.load() << "\n";
            try {
                // Do something
                using namespace std::chrono_literals;
                //std::cout << "Calling next()\n";
                boost::optional<T> r = next();
                if(r && !r.get().isEmpty()) {
                    //std::cout << "r: " << r.get().getFirst() << "\n";
                    mBusy.store(true);
                    auto solution = r.get()();
                    if (solution.first >= 0) {
                        solutions.push_back(r.get()());
                    }
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

    template <typename T, typename ValueType>
    void CLooper<T, ValueType>::abortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        }
    }

    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::poolAbortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        } else {
            return false;
        }
        return true;
    }

    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::solutionsReady() {
        if (solutions.size() == expectedSolutions) {
            return true;
        }
        return false;
    }

    template <typename T, typename ValueType>
    bool CLooper<T, ValueType>::post(T &&aRunnable) {
        if(not running()) {
            // deny insertion
            std::cout << "Looper not running\n";
            return false;
        }

        try {
            std::lock_guard guard(mRunnablesMutex);
            mRunnables.push(std::move(aRunnable));
            expectedSolutions += 1;
        } catch (...) {
            return false;
        }

        return true;
    }

    template <typename T, typename ValueType>
    void CLooperPool<T, ValueType>::stop() {
        // for each running thread stop it
        for (const auto & looper: mLoopers) {
            looper->stop();
        }
    }

    template <typename T, typename ValueType>
    bool CLooperPool<T, ValueType>::run() {
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

    template <typename T, typename ValueType>
    bool CLooperPool<T, ValueType>::running() {
        bool check = true;
        for (const auto & mLooper : mLoopers) {
            if(!mLooper->running()) {
                return false;
            }
        }
        return true;
    }

    template<typename T, typename ValueType>
    void CLooperPool<T, ValueType>::collectSolutions() {
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

    template <typename T, typename ValueType>
    std::vector<std::shared_ptr<typename CLooper<T, ValueType>::CDispatcher>> CLooperPool<T, ValueType>::getDispatchers() {
        std::vector<std::shared_ptr<typename CLooper<T, ValueType>::CDispatcher>> dispatchers;
        for (const auto & mLooper : mLoopers) {
            dispatchers.push_back(mLooper->getDispatcher());
        }
        return dispatchers;
    }

    template <typename T, typename ValueType>
    void CLooperPool<T, ValueType>::solve(std::vector<mythread::SchedulerProblem<ValueType>> tasks) {
        // implementation of the scheduler solver

        auto dispatchers = getDispatchers();
        uint countDown = tasks.size();
        uint k = 0;
        while(countDown > 0) {
            k = k % mLoopers.size();
            mythread::SchedulerProblem task = tasks.back();
            if(dispatchers[k]->post(std::move(task))) {
                //std::cout << "Thread: " << k << " accepted task " << countDown << "\n";
                --countDown;
                tasks.pop_back();
            } else {
                //std::cout << "Thread looper busy or not running\n";
            }
            ++k;
        }

        collectSolutions();
    }

    template <typename T, typename ValueType>
    void CLooperPool<T, ValueType>::solve(std::vector<mythread::DTMCProblem<ValueType>> tasks) {
        // implementation of the DTMC solver
    }

    template<typename T, typename ValueType>
    std::vector<std::pair<uint, double>>& CLooperPool<T, ValueType>::getSolutions() {

        std::sort(solutions.begin(), solutions.end(), [](const auto& left, const auto&right){
            return left.first < right.first;
        });
        return solutions;
    }

    template <typename ValueType>
    void mythread::SchedulerProblem<ValueType>::getProblemData(uint &index_, double &x_, double &y_) {
        index_ = this->index;
        x_ = this->x;
        y_ = this->y;
    }

    template <typename ValueType>
    void mythread::DTMCProblem<ValueType>::getProblemData(uint &index_, double &x_, double &y_) {
        index_ = this->index;
        x_ = this->x;
        y_ = this->y;
    }

}