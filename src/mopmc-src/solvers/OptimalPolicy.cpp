//
// Created by thomas on 3/09/23.
//
#include "OptimalPolicy.h"
#include <storm/models/sparse/Mdp.h>
#include <random>

template<typename SparseModelType>
void mopmc::solver::optimalPolicy(
    Problem<SparseModelType> &problem,
    typename Problem<SparseModelType>::dVec &w) {

    // TODO finish off this value iteration algorithm

    // do one multiplication of the rewards matrix and the weight vector
    if (storm::solver::minimize(problem.dir)) {
        w *= -static_cast<typename SparseModelType::ValueType>(-1.0);
    }

    computeInitialValueVector(problem, w);

    problem.y = problem.model.getRewardMatrix() * w;
    problem.y = problem.model.getTransitionMatrix() * problem.x;

    bool unstable = optimalActions(problem);
    typename SparseModelType::ValueType eps = computeEpsilon(problem);
}

template<typename SparseModelType>
bool mopmc::solver::optimalActions(Problem<SparseModelType> &problem) {
    // TODO check this function works
    // loop over all of the states
    // for a given state find the optimal action in the states
    bool totalUnstable = false;
    auto &rowGroupIndices = problem.model.getRowGroupIndices();
    for (uint_fast64_t state = 0; state < problem.model.getNumberOfStates() - 1; ++state) {
        uint_fast64_t beginAction = rowGroupIndices[state];
        uint_fast64_t endAction = rowGroupIndices[state + 1];

        typename SparseModelType::ValueType minMaxValue =
                std::numeric_limits<typename SparseModelType::ValueType>::min();
        uint_fast64_t maxIndex = -1;
        bool unstable = false;

        for (uint_fast64_t i = beginAction; i <= endAction; ++i) {
            if (problem.y[i] > minMaxValue) {
                unstable = true;
                totalUnstable = true;
                minMaxValue = problem.y[i];
                maxIndex = i;
            }
        }
        if (unstable) {
            problem.pi[state] = maxIndex;
            problem.x[state] = minMaxValue;
        }
    }
    return totalUnstable;
}

template <typename SparseModelType>
typename SparseModelType::ValueType mopmc::solver::computeEpsilon(Problem<SparseModelType> &problem) {
    return (problem.x - problem.xprev).maxCoeff();
}

template <typename SparseModelType>
void mopmc::solver::computeInitialValueVector(
    Problem<SparseModelType> &problem,
    typename Problem<SparseModelType>::dVec &w) {
    // When the value vectors are zero we just choose the action that gives the highest
    // return in a given state
}

// Explicit definitions

template void mopmc::solver::optimalPolicy<storm::models::sparse::Mdp<double>>(
    Problem<storm::models::sparse::Mdp<double>> &problem,
    typename Problem<storm::models::sparse::Mdp<double>>::dVec &w);

template bool mopmc::solver::optimalActions(Problem<storm::models::sparse::Mdp<double>> &problem);

template storm::models::sparse::Mdp<double>::ValueType mopmc::solver::computeEpsilon(
        Problem<storm::models::sparse::Mdp<double>> &problem);

template struct mopmc::solver::Problem<storm::models::sparse::Mdp<double>>;