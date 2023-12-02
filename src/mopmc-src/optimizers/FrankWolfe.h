//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_FRANKWOLFE_H
#define MOPMC_FRANKWOLFE_H

#include <vector>
#include <cassert>
#include <algorithm>
#include <Eigen/Dense>
#include "PolytopeTypeEnum.h"
#include "../convex-functions/BaseConvexFunction.h"
#include "../convex-functions/TotalReLU.h"
#include "LinOpt.h"
#include "LineSearch.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class FrankWolfe {
    public:
        explicit FrankWolfe(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        Vector<V> argmin(std::vector<Vector<V>> &Phi,
                         std::vector<Vector<V>> &W,
                         Vector<V> &xIn,
                         PolytopeType rep,
                         bool doLineSearch);

        Vector<V> argmin(std::vector<Vector<V>> &Phi,
                         Vector<V> &xIn,
                         PolytopeType rep,
                         bool doLineSearch);

        V lineSearch(Vector<V> &vLeft, Vector<V> &vRight, V epsilon2) {
            return 1.0;
        };

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;
        V epsilon = 1e-3;
        V gamma;
        V gamma0 = static_cast<V>(0.1);
        int maxIter = 100;
    };



}


#endif //MOPMC_FRANKWOLFE_H
