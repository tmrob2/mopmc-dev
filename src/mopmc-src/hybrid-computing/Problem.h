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

namespace hybrid {
template<typename ValueType>
class SchedulerProblem {
public:
    //! The problem needs to solve a value iteration problem.
    //! Take the inputs for the value iteration problem
    SchedulerProblem(uint index,
                     Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                     Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x_,
                     std::vector<ValueType> &r_,
                     std::vector<uint64_t> &pi_,
                     std::vector<typename storm::storage::SparseMatrix<ValueType>::indexType> const &rowGroupIndices_)
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

    // SchedulerProblem callable
    std::pair<int, double> operator()() const {
        //using namespace std::chrono_literals;
        //std::this_thread::sleep_for(5s);
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

template<typename ValueType>
class DTMCProblem {
public:
    //! The problem needs to solve a value iteration problem.
    //! Take the inputs for the value iteration problem
    DTMCProblem(uint index,
                     Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                     Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x_,
                     std::vector<ValueType> &r_, // this should change
                     std::vector<uint64_t> &pi_) // this should be const may not be necessary
            : index(index), x(x_), matrix(transitionSystem), r(r_), pi(pi_) {
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
};
}
#endif //MOPMC_PROBLEM_H
