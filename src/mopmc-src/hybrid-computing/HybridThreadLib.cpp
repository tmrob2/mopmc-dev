//
// Created by thomas on 22/10/23.
//
#include "HybridThreadLib.h"
#include <iostream>

namespace mythread {
// Running the infinite loop
    template <typename ValueType>
    bool CLooper<ValueType>::run() {

        try{
            mThread = std::thread(&CLooper::runFunc, this);
        } catch (...) {
            return false;
        }

        return true;
    }

    template <typename ValueType>
    bool CLooper<ValueType>::stop() {
        abortAndJoin();
        return true;
    }

    template <typename ValueType>
    bool CLooper<ValueType>::getAbortRequested() const {
        return mAbortRequested.load();
    }

    template <typename ValueType>
    bool CLooper<ValueType>::running() const {
        return mRunning.load();
    }

    template <typename ValueType>
    bool CLooper<ValueType>::busy() const {
        return mBusy.load();
    }

    template <typename ValueType>
    boost::optional<mythread::Problem<ValueType>> CLooper<ValueType>::next() {
        // A mutex is required to guard against simultaneous access to the task collection
        // by the worker and dispatching threads.
        std::lock_guard guard(mRunnablesMutex); // only works with Cxx17 but we force this standard in the CMAKE

        boost::optional<Problem<ValueType>> problem;
        if(!mRunnables.empty()) {
            problem = mRunnables.front();
            mRunnables.pop();
        }

        return problem;

    }

    //! Empties all of the values in the solution vector
    template <typename ValueType>
    std::vector<std::pair<int, double>> CLooper<ValueType>::getSolution() {
        std::vector<std::pair<int, double>> rtn = {};
        if (!solutions.empty()) {
            solutions.swap(rtn);
        }
        return rtn;
    }

    template <typename ValueType>
    void CLooper<ValueType>::runFunc() {
        mRunning.store(true);

        while (!mAbortRequested.load()) {
            //std::cout << "Abort Requested: " << mAbortRequested.load() << "Running: " << mRunning.load() << "\n";
            try {
                // Do something
                using namespace std::chrono_literals;
                //std::cout << "Calling next()\n";
                boost::optional<Problem<ValueType>> r = next();
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

    template <typename ValueType>
    void CLooper<ValueType>::abortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        }
    }

    template <typename ValueType>
    bool CLooper<ValueType>::poolAbortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        } else {
            return false;
        }
        return true;
    }

    template <typename ValueType>
    bool CLooper<ValueType>::solutionsReady() {
        if (solutions.size() == expectedSolutions) {
            return true;
        }
        return false;
    }

    template <typename ValueType>
    bool CLooper<ValueType>::post(mythread::Problem<ValueType> &&aRunnable) {
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

    template <typename ValueType>
    void CLooperPool<ValueType>::stop() {
        // for each running thread stop it
        for (const auto & looper: mLoopers) {
            looper->stop();
        }
    }

    template <typename ValueType>
    bool CLooperPool<ValueType>::run() {
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

    template <typename ValueType>
    bool CLooperPool<ValueType>::running() {
        bool check = true;
        for (const auto & mLooper : mLoopers) {
            if(!mLooper->running()) {
                return false;
            }
        }
        return true;
    }

    template<typename ValueType>
    void CLooperPool<ValueType>::collectSolutions() {
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

    template <typename ValueType>
    std::vector<std::shared_ptr<typename CLooper<ValueType>::CDispatcher>> CLooperPool<ValueType>::getDispatchers() {
        std::vector<std::shared_ptr<typename CLooper<ValueType>::CDispatcher>> dispatchers;
        for (const auto & mLooper : mLoopers) {
            dispatchers.push_back(mLooper->getDispatcher());
        }
        return dispatchers;
    }

    template <typename ValueType>
    void CLooperPool<ValueType>::solve(std::vector<Problem<ValueType>> tasks) {
        // assign the tasks to free threads
        // assumes that all loopers are running.

        //std::cout << "Threads started\n";

        auto dispatchers = getDispatchers();
        uint countDown = tasks.size();
        uint k = 0;
        while(countDown > 0) {
            k = k % mLoopers.size();
            mythread::Problem task = tasks.back();
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

    template<typename ValueType>
    std::vector<std::pair<uint, double>>& CLooperPool<ValueType>::getSolutions() {

        std::sort(solutions.begin(), solutions.end(), [](const auto& left, const auto&right){
            return left.first < right.first;
        });
        return solutions;
    }

    template <typename ValueType>
    void mythread::Problem<ValueType>::getProblemData(uint &index_, double &x_, double &y_) {
        index_ = this->index;
        x_ = this->x;
        y_ = this->y;
    }

}