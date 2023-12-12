//
// Created by thomas on 7/09/23.
//

#ifndef MOPMC_INDUCEDDTMCSOLVER_H
#define MOPMC_INDUCEDDTMCSOLVER_H

#include <storm/utility/constants.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>


namespace mopmc {
namespace solver {
namespace linsystem {

template<typename ValueType>
void solverHelper(Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &b,
                  Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &matrix,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &identity);

template<typename ValueType>
void solverHelper(Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &b,
                  Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>> &x,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &matrix,
                  Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &id);
}
}
}
#endif //MOPMC_INDUCEDDTMCSOLVER_H
