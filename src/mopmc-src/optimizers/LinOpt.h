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

        int findOptimalSeparatingDirection(std::vector<Vector<V>> &Vertices,
                                           PolytopeType &polytopeType,
                                           Vector<V> &gradient,
                                           Vector<V> &sign,
                                           Vector<V> &optimalDirection);

        int checkPointInConvexHull(const std::vector<Vector<V>> &Vertices,
                                   const Vector<V> &point,
                                   int &feasible);

        int optimizeVtx(const std::vector<Vector<V>> &Vertices,
                        PolytopeType &polytopeType,
                        Vector<V> &gradient,
                        Vector<V> &point);

        int optimizeHlsp(const std::vector<Vector<V>> &Vertices,
                         const std::vector<Vector<V>> &Weights,
                         PolytopeType &polytopeType,
                         Vector<V> &gradient,
                         Vector<V> &point);

        int findMaximumFeasibleStep(const std::vector<Vector<V>> &Vertices,
                                    const Vector<V> &gradient,
                                    Vector<V> point, V step);

    };
}

#endif //MOPMC_LINOPT_H
