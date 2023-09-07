//
// Created by thomas on 7/09/23.
//

#include "ValueIteration.h"

template <typename ValueType>
void valueIteration(Eigen::SparseMatrix<ValueType, Eigen::RowMajor> &transitionSystem,
                    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &x,
                    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &r){
    // Instantiate y and xprev
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> xprev = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>::Zero(x.size());
    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> y = Eigen::Matrix<ValueType, Eigen::Dynamic, 1>::Zero(transitionSystem.rows());

    // compute y = r + P.x

    y = r;
    y += transitionSystem * x;


};


