//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_FRANKWOLFE_H
#define MOPMC_FRANKWOLFE_H

#include <vector>
#include <set>
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

    enum FWOptimizationMethod {
        LP, AWAY_STEP
    };

    template<typename V>
    class FrankWolfe {
    public:
        explicit FrankWolfe(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        FrankWolfe(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                   uint64_t maxSize);

        Vector<V> argmin(std::vector<Vector<V>> &Phi,
                         std::vector<Vector<V>> &W,
                         Vector<V> &xIn,
                         PolytopeType polytopeType,
                         bool doLineSearch=true);

        Vector<V> argmin(std::vector<Vector<V>> &Phi,
                         Vector<V> &xIn,
                         PolytopeType polytopeType,
                         bool doLineSearch=true);

        Vector<V> argminByAwayStep(std::vector<Vector<V>> &Phi,
                                   Vector<V> &xIn,
                                   bool doLineSearch=true);

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;

        Vector<V> alpha;
        std::set<V> S;

    };



}


#endif //MOPMC_FRANKWOLFE_H
