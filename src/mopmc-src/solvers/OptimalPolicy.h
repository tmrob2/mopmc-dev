//
// Created by thomas on 3/09/23.
//

#ifndef MOPMC_OPTIMALPOLICY_H
#define MOPMC_OPTIMALPOLICY_H

#include <storm/utility/constants.h>
#include "../SparseModel2.h"
#include <Eigen/Sparse>
#include <vector>
#include <storm/storage/BitVector.h>
#include <storm/solver/OptimizationDirection.h>

namespace mopmc {
namespace solver {

template <typename SparseModelType>
struct Problem {
    typedef Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> dVec;

    mopmc::sparsemodel::SparseModelBuilder<SparseModelType>& model;
    typename SparseModelType::ValueType epsilon;
    dVec& x;
    dVec& xprev;
    dVec& y;
    std::vector<uint_fast64_t>& pi;
    storm::solver::OptimizationDirection dir;

    Problem(mopmc::sparsemodel::SparseModelBuilder<SparseModelType>& model,
        typename SparseModelType::ValueType epsilon,
        dVec& x,
        dVec& y,
        std::vector<uint_fast64_t>& pi,
        storm::solver::OptimizationDirection dir)
        : model(model), epsilon(epsilon), x(x), xprev(x), y(y), pi(pi), dir(dir) {
        // Intentionally left blank
    }
};

template<typename SparseModelType>
void computeInitialValueVector(Problem<SparseModelType> &problem, typename Problem<SparseModelType>::dVec &w);

template<typename SparseModelType>
void optimalPolicy(Problem<SparseModelType> &problem, typename Problem<SparseModelType>::dVec &w);

template<typename SparseModelType>
bool optimalActions(Problem<SparseModelType>& problem);

template<typename SparseModelType>
typename SparseModelType::ValueType computeEpsilon(Problem<SparseModelType>& problem);
}
}

#endif //MOPMC_OPTIMALPOLICY_H
