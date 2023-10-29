//
// Created by thomas on 22/10/23.
//

#ifndef MOPMC_HYBRIDTHREADLIB_H
#define MOPMC_HYBRIDTHREADLIB_H

#include <thread>
#include <atomic>
#include <memory>
#include <functional>
#include <stdexcept>
#include <queue>
#include <mutex>
#include <storm/utility/constants.h>
#include <Eigen/Sparse>
#include <boost/optional.hpp>
#include <storm/storage/SparseMatrix.h>

namespace mythread {

// Forward declaration
template <typename ValueType>
class Problem;


/*!
 * Loopers are objects which contain or are attached to a thread with a conditional infinite loop.
 * This loop runs as long as the abort-criteria is unmet. Within this loop, arbitrary actions can be
 * performed
 *
 * Implements start, run stop
 *
 * Define a CLooper class, which contains std::thread-member and a run-method which will create
 * the thread invoking runFunc - the second method - implementing an effective thread operation.
 *
 * Stopping the loop:
 *  In order to stop the looper, add an abort criteria to the infinite loop -mAbortRequested
 *  of type std::atomic<bool> which is checked against each iteration.
 *
 *  Add in a private method abortAndJoin() which will set the mAbortRequested flag to true -
 *  invokes join() on the thread and waits until the looper-function has been exited and the worker
 *  thread was joined.
 *
 *  The destructor will invoke abortAndJoin() iun the case the looper goes out of scope.
 *
 * Tasks:
 *  In the context of loopers, tasks are executable portions of code sharing a common signature
 *  i.e. one or more Tasks, which can be fetched from an internal collection (e.g. FIFO queue)
 *  and can be executed by the worker thread
 *
 * Dispatchers:
 *  Tasks are pushed to the queue with dispatching.
 *  A dispatcher will accept a task but will manage the insertion into the working-queue
 *  This way some fancy usage scenarios can be handled such as delayed execution or immediate
 *  posting.
 */

/*
 * std::function<void()> is a type that represents a callable object (function, lambda function,
 * functor) that takes no arguments and returns `void` i.e. does not return anything.
 *
 * This is often used to encapsulate and store functions or function-like objects with this
 * signature. The question is, is this useful?
 */
using Runnable = std::function<void()>;

template <typename ValueType>
class CLooper {
public:
    CLooper(uint id) : id(id), mRunning(false), mAbortRequested(false), mRunnables(),
                       mRunnablesMutex(), mDispatcher(std::shared_ptr<CDispatcher>(new CDispatcher(*this))),
                       mBusy(false), expectedSolutions(0) {

    };
    // Copy denied, move to be implemented

    ~CLooper() {
        // called in case the looper goes out of scope.
        abortAndJoin();
    }

    // To be called once the looper should start looping
    bool run();

    // To be called to stop a thread
    bool stop();

    // Check if the thread is running
    bool running() const;

    // Check if the thread is busy computing
    bool busy() const;

    // Check if poison pill inserted
    bool getAbortRequested() const;

    // Return solutions from the thread
    std::vector<std::pair<int, double>> getSolution();

    // Computes the next problem
    boost::optional<Problem<ValueType>> next();

    // Flag to check if all tasks have been computed by the thread
    bool solutionsReady();

    // Task Dispatcher
    class CDispatcher {
        friend class CLooper; // Allow the dispatcher access to the private members

    public:

        bool post(Problem<ValueType> &&aRunnable) {
            return mAssignedLooper.post(std::move(aRunnable));
        }

    private:
        explicit CDispatcher(CLooper &aLooper) : mAssignedLooper(aLooper) {};

        CLooper &mAssignedLooper;
    };

    std::shared_ptr<CDispatcher> getDispatcher() { return mDispatcher; };

    bool poolAbortAndJoin();

private:

    void abortAndJoin();

    // Implements a thread function
    void runFunc();

    bool post(Problem<ValueType> &&aRunnable);


    std::thread mThread;
    std::atomic_bool mRunning;
    std::atomic_bool mBusy;
    std::atomic_bool mAbortRequested;
    std::queue<Problem<ValueType>> mRunnables; /* This is just a standard queue it can take anything
                                  * probably the best thing to do is insert a class
                                  * The class could evn be a functor which calls its own
                                  * model checking operation
                                  */
    std::recursive_mutex mRunnablesMutex;
    std::shared_ptr<CDispatcher> mDispatcher;
    std::vector<std::pair<int, double>> solutions;
    uint expectedSolutions;
    uint id;
};

template <typename ValueType>
class Problem {
public:
    //! The problem needs to solve a value iteration problem.
    //! Take the inputs for the value iteration problem
    Problem(uint index,
            Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
            Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x_,
            std::vector<ValueType> &r_,
            std::vector<uint64_t> &pi_,
            std::vector<typename storm::storage::SparseMatrix<ValueType>::indexType> const& rowGroupIndices_)
            : index(index), x(x_), matrix(transitionSystem), r(r_), pi(pi_), rowGroupIndices(rowGroupIndices_) {
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

    void getProblemData(uint &index_, double &x_, double &y_);

    std::pair<int, double> operator()() const {
        using namespace std::chrono_literals;
        //std::this_thread::sleep_for(5s);
        std::pair<uint, double> sol;
        sol.first = index;
        sol.second = 0.;
        return sol;
    }

private:
    uint index;
    Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &matrix;
    Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x;
    std::vector<ValueType> &r;
    std::vector<uint64_t> &pi;
    std::vector<typename storm::storage::SparseMatrix<ValueType>::indexType> const& rowGroupIndices;
};

template <typename ValueType>
class CLooperPool {
public:
    CLooperPool(std::vector<std::unique_ptr<CLooper<ValueType>>>&& loopers): mLoopers(std::move(loopers)){};

    ~CLooperPool() {
        stop();
    }

    bool run();

    bool running();

    void stop();

    void solve(std::vector<Problem<ValueType>> tasks);

    void collectSolutions();

    std::vector<std::pair<uint, double>>& getSolutions();

    std::vector<std::shared_ptr<typename CLooper<ValueType>::CDispatcher>> getDispatchers();

private:
    std::vector<std::unique_ptr<CLooper<ValueType>>>&& mLoopers;
    std::vector<std::pair<uint, double>> solutions;
};
}
#endif //MOPMC_HYBRIDTHREADLIB_H
