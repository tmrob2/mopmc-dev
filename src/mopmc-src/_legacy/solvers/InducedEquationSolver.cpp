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
    std::cout << "id: dim(" << id.rows() << "," <<  id.cols() << ")\n";
    std::cout << "A: dim(" << matrix.rows() << "," <<  matrix.cols() << ")\n";


    Eigen::SparseMatrix<ValueType, Eigen::RowMajor> Z = id - matrix;


    Eigen::BiCGSTAB<Eigen::SparseMatrix<ValueType, Eigen::RowMajor>> solver;
    solver.compute(Z);
    x = solver.solve(b);
    std::cout << "Estimated error: " << solver.error() << "\n";
};

template<typename ValueType>
void solverHelper(Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &b,
                  Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &matrix,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &id){
    //
    std::cout << "id: dim(" << id.rows() << "," <<  id.cols() << ")\n";
    std::cout << "A: dim(" << matrix.rows() << "," <<  matrix.cols() << ")\n";


    Eigen::SparseMatrix<ValueType, Eigen::RowMajor> Z = id - matrix;


    Eigen::BiCGSTAB<Eigen::SparseMatrix<ValueType, Eigen::RowMajor>> solver;
    solver.compute(Z);
    x = solver.solve(b);
    std::cout << "Estimated error: " << solver.error() << "\n";
};

template void solverHelper(Eigen::Matrix<double, Eigen::Dynamic, 1> &b,
                           Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> &x,
                           Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix,
                           Eigen::SparseMatrix<double, Eigen::RowMajor> &id);

template void solverHelper(Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> &b,
                  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> &x,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> &id);

}
}
}
