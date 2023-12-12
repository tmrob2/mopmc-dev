//
// Created by thomas on 7/09/23.
//

#include "IterativeSolver.h"
#include "InducedEquationSolver.h"
#include <storm/models/sparse/Mdp.h>
#include <Eigen/IterativeLinearSolvers>
#include <storm/solver/SolverStatus.h>

namespace mopmc {
namespace solver::iter {
template <typename ValueType>
void valueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                    Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                    std::vector<ValueType> &r,
                    std::vector<uint64_t> &pi,
                    std::vector<typename storm::storage::SparseMatrix<ValueType>::index_type> const& rowGroupIndices){
    // Instantiate y and xprev
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> xprev = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>::Zero(x.size());
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> y = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>::Zero(r.size());
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> R(r.size());
    for(uint_fast64_t i = 0; i < r.size(); ++i) {
        if (i < x.size()) {
            xprev[i] = x[i];
        }
        R(i) = r[i];
    }

    ValueType eps = std::numeric_limits<ValueType>::max();
    // compute y = r + P.x
    Eigen::Index maxRow;
    uint_fast64_t iterations = 0;

   //std::cout << "Actions in initial state: " << rowGroupIndices[0] << " - " <<  rowGroupIndices[1] << "\n";

    do {
        y = R;

        y += transitionSystem * x;

        // compute the new x
        nextBestPolicy(y, x, pi, rowGroupIndices);
        // compute the error
        eps = computeEpsilon(x, xprev, maxRow);
        xprev = x;
        //std::cout << "Eps[" << iterations << "]: " << eps << "\n";
        ++iterations;
    } while (eps > 1e-6);
}

template <typename ValueType>
void objValueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                       Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> &x,
                       Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> &R){
    // Instantiate y and xprev
    Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> y =
            Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>::Zero(R.rows(), R.cols());
    ValueType eps = std::numeric_limits<ValueType>::max();
    // compute Y = Y + P.X
    uint_fast64_t iterations = 0;
    //std::cout << transitionSystem.toDense() << std::endl;
    do {
        y = R;

        y += transitionSystem * x;

        //std::cout << y.transpose() << "\n\n";

        //std::cout << y << std::endl;
        eps = computeEpsilonMatrix(x, y);
        /*for(uint_fast64_t i = 0; i < x.rows(); ++i) {
            std::cout << x(i) << " " << y(i) << "\n";
        }*/
        //std::cout << "Eps[" << iterations << "]: " << eps << "\n";
        x = y;
        if (iterations > 10000) {
            break;
        }
        ++iterations;
    } while (eps > 1e-6);
}

template <typename ValueType>
bool nextBestPolicy(Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &y,
                    Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                    std::vector<uint64_t> &pi,
                    std::vector<typename storm::storage::SparseMatrix<ValueType>::index_type> const& rowGroupIndices) {

    for (uint_fast64_t state = 0; state < x.size(); ++state) {
        uint_fast64_t actionBegin = rowGroupIndices[state];
        uint_fast64_t actionEnd = rowGroupIndices[state+1];
        ValueType maxValue = x[state];
        uint64_t maxIndex = pi[state];
        for (uint_fast64_t action = 0; action < (actionEnd - actionBegin); ++action) {
            if (y[actionBegin + action] > maxValue) {
                maxIndex = action;
                maxValue = y[actionBegin+action];
            }
        }
        //std::cout << "x' " << maxValue << " xold: " << x[state]<<"\n";
        x[state] = maxValue;
        pi[state] = maxIndex;
    }
    //std::cout << "\n\n";
    return true;
}

template <typename ValueType>
ValueType computeEpsilon(Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                         Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &xprev,
                         Eigen::Index &maxRow){
     return (x - xprev).maxCoeff(&maxRow);
}

template <typename ValueType>
ValueType computeEpsilonMatrix(Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> &x,
                               Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> &y){
    auto z = y - x;
    /*double maxElement = std::numeric_limits<double>::min();
    for(uint_fast64_t i = 0; i < z.rows(); ++i) {
        if(z(i) > maxElement) {
            maxElement = z(i);
        }
    }
    return maxElement;*/
    return z.maxCoeff();
}

template <typename SparseModelType, typename ValueType>
void policyIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                     Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &identity,
                     Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                     std::vector<ValueType> &r, std::vector<uint64_t> &pi,
                     std::vector<typename storm::storage::SparseMatrix<ValueType>::index_type> const& rowGroupIndices){
    Eigen::BiCGSTAB<Eigen::SparseMatrix<ValueType, Eigen::RowMajor>> solver;
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> subB = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>::Zero(x.size());
    storm::solver::SolverStatus status = storm::solver::SolverStatus::InProgress;
    uint64_t iterations = 0;
    bool schedulerImproved = false;
    // solve the equation for the DTMC
    //Eigen::SparseMatrix<ValueType, Eigen::RowMajor> subMatrix = mopmc::solver::linsystem::eigenInducedTransitionMatrix<SparseModelType>(
    //        transitionSystem, b, subB, pi);
    //mopmc::solver::linsystem::solverHelper(subB, x, subMatrix, identity);
    for (uint_fast64_t group = 0; group < x.size(); ++group){
        uint_fast64_t currentChoice = pi[group];
        for (uint_fast64_t choice = rowGroupIndices[group]; choice < rowGroupIndices[group+1]; ++choice){
            if (choice == currentChoice) {
                continue;
            }

            // Create the value of the choice
            auto choiceValue = storm::utility::zero<ValueType>();
            typename Eigen::SparseMatrix<ValueType, Eigen::RowMajor>::InnerIterator it(transitionSystem, choice);
            for(;it;++it){
                choiceValue += it.value() * x[it.col()];
            }
            choiceValue += r[choice];

            if (choiceValue > x[group]) {
                schedulerImproved = true;
                pi[group] = choice;
                x[group] = std::move(choiceValue);
            }
        }
    }
    if (!schedulerImproved) {
        // then we are finished
        status = storm::solver::SolverStatus::Converged;
    } else {
        std::cout << "Scheduler improved\n";
    }

    ++iterations;
}

// Explicit instantiation
template void valueIteration(Eigen::SparseMatrix<double, Eigen::RowMajor> &transitionSystem,
                             Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> &x,
                             std::vector<double> &r,
                             std::vector<uint64_t> &pi,
                             std::vector<typename storm::storage::SparseMatrix<double>::index_type> const& rowGroupIndices);

template void objValueIteration(Eigen::SparseMatrix<double, Eigen::RowMajor> &transitionSystem,
                                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& x,
                                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& R);

template void policyIteration<storm::models::sparse::Mdp<double>>(
    Eigen::SparseMatrix<double, Eigen::RowMajor> &transitionSystem,
    Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix,
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> &x,
    std::vector<double> &r, std::vector<uint64_t> &pi,
    std::vector<typename storm::storage::SparseMatrix<double>::index_type> const& rowGroupIndices);
}
}
