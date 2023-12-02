//
// Created by thomas on 05/11/23.
//

#ifndef MOPMC_PROBLEM_H
#define MOPMC_PROBLEM_H


#include <cstdint>
#include <storm/storage/SparseMatrix.h>
#include <storm/utility/constants.h>
#include <Eigen/Sparse>
#include "mopmc-src/solvers/CuVISolver.h"
#include "Looper.h"

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
class SchedulerProblem {
public:
    //! The problem needs to solve a value iteration problem.
    //! Take the inputs for the value iteration problem
    SchedulerProblem(uint index, std::vector<ValueType> w, ThreadSpecialisation spec) {
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

    // SchedulerProblem callable
    std::pair<int, double> operator()() const {
        //using namespace std::chrono_literals;
        //std::this_thread::sleep_for(5s);

        // TODO if the thread is a GPU specialisation then send the data to the CPU for computation
        //   here, which will involve launching some kernel

        // TODO another specialisation is the type of problem to solve, is it a DTMC problem
        //  or is it a scheduler optimisation problem

        // A key question at this point is what data does the Problem have access to, and where does it
        // live. 
        std::pair<uint, double> sol;
        sol.first = index;
        // run value iteration on a problem
        sol.second = 0.;
        return sol;
    }

private:
    uint index;
    Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &matrix;
    Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x;
    std::vector<ValueType> &r;
    std::vector<uint64_t> &pi;
    std::vector<typename storm::storage::SparseMatrix<ValueType>::indexType> const &rowGroupIndices;
};
}
#endif //MOPMC_PROBLEM_H
