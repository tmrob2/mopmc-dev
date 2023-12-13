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
#include "BaseOptimizer.h"
#include "PolytopeTypeEnum.h"
#include "../convex-functions/BaseConvexFunction.h"
#include "../convex-functions/TotalReLU.h"
#include "LinOpt.h"
#include "LineSearch.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    enum FWOption {
        LINOPT, AWAY_STEP, BLENDED, BLENDED_STEP_OPT
    };

    template<typename V>
    class FrankWolfe : public BaseOptimizer<V>{
    public:
        explicit FrankWolfe(FWOption optMethod, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        FrankWolfe(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                   uint64_t maxSize);

        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) override;

        FWOption fwOption{};
        Vector<V> alpha;
        std::set<V> activeSet;
        bool lineSearch{true};

    private:
        Vector<V> argminByLinOpt(const std::vector<Vector<V>> &Phi,
                                 const std::vector<Vector<V>> &W,
                                 Vector<V> &xIn,
                                 PolytopeType polytopeType,
                                 bool doLineSearch= true);

        Vector<V> argminByLinOpt(const std::vector<Vector<V>> &Phi,
                                 Vector<V> &xIn,
                                 PolytopeType polytopeType,
                                 bool doLineSearch= true);

        Vector<V> argminByAwayStep(const std::vector<Vector<V>> &Phi,
                                   bool doLineSearch=true);

        Vector<V> argminByBlendedGD(const std::vector<Vector<V>> &Vertices,
                                    bool doLineSearch=true,
                                    bool feasibilityCheckOnly=true);
    };



}


#endif //MOPMC_FRANKWOLFE_H
