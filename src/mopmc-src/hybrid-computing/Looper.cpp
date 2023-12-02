//
// Created by thomas on 22/10/23.
//
#include "Looper.h"
#include <iostream>
#include <storm/adapters/EigenAdapter.h>

namespace hybrid {

    template<typename M, typename V>
    CLooper<M, V>::CLooper(
            uint id_, hybrid::ThreadSpecialisation spec, PrepReturnType const& model,
            std::vector<int> const& rowGroupIndices)
            : id(id_), mRunning(false), mAbortRequested(false), mRunnables(), mRunnablesMutex(),
              mDispatcher(std::shared_ptr<CDispatcher>(new CDispatcher(*this))),
              mBusy(false), expectedSolutions(0) {
        m = model.objectives.size(); // number of objectives
        n = model.preprocessedModel->getNumberOfChoices();
        k = model.preprocessedModel->getNumberOfStates();

        {
            // kill rho at end of block
            std::vector<std::vector<V>> rho(m);
            std::vector<V> rhoFlat(n * m);

            for (uint_fast64_t i = 0; i < m; ++i) {
                auto& name = model.objectives[i].formula->asRewardOperatorFormula().getRewardModelName();
                rho[i] = model.preprocessedModel->getRewardModel(name)
                        .getTotalRewardVector(model.preprocessedModel->getTransitionMatrix());
                for(uint_fast64_t j = 0; j < n; ++j) {
                    rhoFlat[i * n + j] = rho[i][j];
                }
            }
        }

        P = storm::adapters::EigenAdapter::toEigenSparseMatrix(
                model.preprocessedModel->getTransitionMatrix());

        P->makeCompressed();
        pi.resize(k, static_cast<V>(0.0));

        switch (spec) {
            case ThreadSpecialisation::CPU:
                // we don't need to do anything here, already a CPU thread
                break;
            case ThreadSpecialisation::GPU:
                // send the data to the GPU
                sendDataGPU(P, rowGroupIndices);
                break;
        }

    }

    // Running the infinite loop
    template <typename M, typename V>
    bool CLooper<M, V>::run() {

        try{
            mThread = std::thread(&CLooper::runFunc, this);
        } catch (...) {
            return false;
        }

        return true;
    }

    template <typename M, typename V>
    bool CLooper<M, V>::stop() {
        abortAndJoin();
        return true;
    }

    template <typename M, typename V>
    bool CLooper<M, V>::getAbortRequested() const {
        return mAbortRequested.load();
    }

    template <typename M, typename V>
    bool CLooper<M, V>::running() const {
        return mRunning.load();
    }

    template <typename M, typename V>
    bool CLooper<M, V>::busy() const {
        return mBusy.load();
    }

    template <typename M, typename V>
    boost::optional<typename CLooper<M, V>::Sch> CLooper<M, V>::next() {
        // A mutex is required to guard against simultaneous access to the task collection
        // by the worker and dispatching threads.
        std::lock_guard guard(mRunnablesMutex); // only works with Cxx17 but we force this standard in the CMAKE

        boost::optional<typename CLooper<M, V>::Sch> problem;
        if(!mRunnables.empty()) {
            problem = mRunnables.front();
            mRunnables.pop();
        }

        return problem;

    }

    // explicit instantiations of send data overloaded
    template <typename M, typename V>
    void sendDataGPU(typename CLooper<M, V>::SpMat& matrix,
                     std::vector<int> const& rowGroupIndices,
                     std::vector<int>& pi) {
        // allocate a matrix
        cuTransitionMatrix (matrix, rowGroupIndices);
    }

    template <typename M, typename V>
    void sendDataCPU(typename CLooper<M, V>::SpMat& matrix) {
        // let this thread take ownership of the transition matrix for CPU operations
    }

    //! Empties all of the values in the solution vector
    template <typename M, typename V>
    std::vector<std::pair<int, double>> CLooper<M, V>::getSolution() {
        std::vector<std::pair<int, double>> rtn = {};
        if (!solutions.empty()) {
            solutions.swap(rtn);
        }
        return rtn;
    }

    template <typename M, typename V>
    void CLooper<M, V>::runFunc() {
        mRunning.store(true);

        while (!mAbortRequested.load()) {
            //std::cout << "Abort Requested: " << mAbortRequested.load() << "Running: " << mRunning.load() << "\n";
            try {
                // Do something
                using namespace std::chrono_literals;
                //std::cout << "Calling next()\n";
                boost::optional<M> r = next();
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

    template <typename M, typename V>
    void CLooper<M, V>::abortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        }
    }

    template <typename M, typename V>
    bool CLooper<M, V>::poolAbortAndJoin() {
        mAbortRequested.store(true);
        if (mThread.joinable()) {
            mThread.join();
        } else {
            return false;
        }
        return true;
    }

    template <typename M, typename V>
    bool CLooper<M, V>::solutionsReady() {
        if (solutions.size() == expectedSolutions) {
            return true;
        }
        return false;
    }

    template <typename M, typename V>
    bool CLooper<M, V>::post(typename CLooper<M, V>::Sch &&aRunnable) {
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

    template <typename M, typename V>
    void CLooperPool<M, V>::stop() {
        // for each running thread stop it
        for (const auto & looper: mLoopers) {
            looper->stop();
        }
    }

    template <typename M, typename V>
    bool CLooperPool<M, V>::run() {
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

    template <typename M, typename V>
    bool CLooperPool<M, V>::running() {
        bool check = true;
        for (const auto & mLooper : mLoopers) {
            if(!mLooper->running()) {
                return false;
            }
        }
        return true;
    }

    template<typename M, typename V>
    void CLooperPool<M, V>::collectSolutions() {
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

    template <typename M, typename V>
    std::vector<std::shared_ptr<typename CLooper<M, V>::CDispatcher>> CLooperPool<M, V>::getDispatchers() {
        std::vector<std::shared_ptr<typename CLooper<M, V>::CDispatcher>> dispatchers;
        for (const auto & mLooper : mLoopers) {
            dispatchers.push_back(mLooper->getDispatcher());
        }
        return dispatchers;
    }

    template <typename M, typename V>
    void CLooperPool<M, V>::solve(std::vector<hybrid::SchedulerProblem<V>> tasks) {
        // implementation of the scheduler solver

        auto dispatchers = getDispatchers();
        uint countDown = tasks.size();
        uint k = 0;
        while(countDown > 0) {
            k = k % mLoopers.size();
            hybrid::SchedulerProblem task = tasks.back();
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

    template<typename M, typename V>
    std::vector<std::pair<uint, double>>& CLooperPool<M, V>::getSolutions() {

        std::sort(solutions.begin(), solutions.end(), [](const auto& left, const auto&right){
            return left.first < right.first;
        });
        return solutions;
    }

    template <typename V>
    void hybrid::SchedulerProblem<V>::getProblemData(uint &index_, double &x_, double &y_) {
        index_ = this->index;
        x_ = this->x;
        y_ = this->y;
    }

}