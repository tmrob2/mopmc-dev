//
// Created by thomas on 7/09/23.
//

#ifndef MOPMC_VALUEITERATION_H
#define MOPMC_VALUEITERATION_H

#include <Eigen/Sparse>

namespace mopmc {
namespace solver::vi{

template <typename ValueType>
void valueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &x,
                    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &r);


template <typename ValueType>
void nextBestPolicy(Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &y,
                    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &x,
                    std::vector<uint64_t> &pi);

template <typename ValueType>
void computeEpsilon(Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &x,
                    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &xprev);


}
}


#endif //MOPMC_VALUEITERATION_H
