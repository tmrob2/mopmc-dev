//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_LINOPT_H
#define MOPMC_LINOPT_H

#include "PolytopeTypeEnum.h"
#include <vector>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <Eigen/Dense>
#include <iostream>
#include "lp_lib.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class LinOpt {
    public:

        int findOptimalSeparatingDirection(std::vector<Vector<V>> &Phi,
                                           PolytopeType &rep,
                                           Vector<V> &d,
                                           Vector<V> &sgn,
                                           Vector<V> &optimalDirection);

        int checkFeasibility(std::vector<Vector<V>> &Phi,
                             Vector<V> &newPoint,
                             bool &feasible);

        int optimizeVtx(const std::vector<Vector<V>> &Phi,
                        PolytopeType &rep,
                        Vector<V> &d,
                        Vector<V> &optimalPoint);

        int optimizeHlsp(const std::vector<Vector<V>> &Phi,
                         const std::vector<Vector<V>> &W,
                         PolytopeType &rep,
                         Vector<V> &d,
                         Vector<V> &optimalPoint);

    };
}

#endif //MOPMC_LINOPT_H
