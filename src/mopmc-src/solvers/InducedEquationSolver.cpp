//
// Created by thomas on 7/09/23.
//

#include "InducedEquationSolver.h"
#include <Eigen/IterativeLinearSolvers>
#include <storm/models/sparse/Mdp.h>
#include <iostream>

namespace mopmc {
namespace solver {
namespace linsystem {

template<typename ValueType>
void solverHelper(Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &b,
                  Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &matrix,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &id){
    //
    std::cout << "id: dim(" << id.rows() << "," <<  id.cols() << "\n";
    std::cout << "A: dim(" << matrix.rows() << "," <<  matrix.cols() << "\n";


    Eigen::SparseMatrix<ValueType, Eigen::RowMajor> Z = id - matrix;


    Eigen::BiCGSTAB<Eigen::SparseMatrix<ValueType, Eigen::RowMajor>> solver;
    solver.compute(Z);
    x = solver.solve(b);
    std::cout << "Estimated error: " << solver.error() << "\n";
};

template <typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> eigenInducedTransitionMatrix(
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> &fullTransitionSystem,
        Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> &b,
        Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1> &subB,
        std::vector<uint64_t>& scheduler) {

    assert(scheduler.size() == fullTransitionSystem.cols());
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> subMatrix(scheduler.size(), scheduler.size());
    for(uint_fast64_t state = 0; state < fullTransitionSystem.cols(); ++state) {
        typename Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>::InnerIterator it(fullTransitionSystem, scheduler[state]);
        for (; it; ++it) {
            subMatrix.insert(state, it.col()) = it.value();
        }
        subB[state] = b[scheduler[state]];
    }

    subMatrix.makeCompressed();
    return subMatrix;
}

template void solverHelper(Eigen::Matrix<double, Eigen::Dynamic, 1> &b,
                           Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> &x,
                           Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix,
                           Eigen::SparseMatrix<double, Eigen::RowMajor> &id);

template Eigen::SparseMatrix<storm::models::sparse::Mdp<double>::ValueType, Eigen::RowMajor> eigenInducedTransitionMatrix<storm::models::sparse::Mdp<double>>(
        Eigen::SparseMatrix<storm::models::sparse::Mdp<double>::ValueType, Eigen::RowMajor> &fullTransitionSystem,
        Eigen::Map<Eigen::Matrix<storm::models::sparse::Mdp<double>::ValueType, Eigen::Dynamic, 1>> &b,
        Eigen::Matrix<storm::models::sparse::Mdp<double>::ValueType, Eigen::Dynamic, 1> &subB,
        std::vector<uint64_t>& scheduler);

}
}
}
