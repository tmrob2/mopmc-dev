//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_FRANKWOLFE_H
#define MOPMC_FRANKWOLFE_H

#include "../convex-functions/BaseConvexFunction.h"
#include "../convex-functions/TotalReLU.h"
#include "BaseOptimizer.h"
#include "LinOpt.h"
#include "LineSearch.h"
#include "PolytopeTypeEnum.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <set>
#include <vector>

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    enum FWOption {
        SIMPLEX_GD,
        LINOPT,
        AWAY_STEP,
        BLENDED,
        BLENDED_STEP_OPT
    };

    template<typename V>
    class FrankWolfe : public BaseOptimizer<V> {
    public:
        explicit FrankWolfe() = default;

        explicit FrankWolfe(FWOption optMethod, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) override;

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearcher<V> lineSearcher;
        FWOption fwOption{};
        Vector<V> alpha;
        std::set<uint64_t> activeSet;
        //bool doLineSearch{true};

    private:
        Vector<V> argmin(const std::vector<Vector<V>> &Vertices);
        void updateWithSimplexGradientDescent(const std::vector<Vector<V>> &Vertices);
        void updateWithForwardOrAwayStep();
        void computeForwardStepIndexAndVector(const std::vector<Vector<V>> &Vertices);
        void computeAwayStepIndexAndVector(const std::vector<Vector<V>> &Vertices);
        void initialize(const std::vector<Vector<V>> &Vertices);

        const V tolerance{1.e-8}, toleranceCosine = std::cos(90.01 / 180. * M_PI);
        const V scale1{0.5}, scale2{0.5};
        int64_t dimension{}, size{};
        Vector<V> xCurrent, xNew, xNewEx, dXCurrent;
        const uint64_t maxIter = 1e2;
        V gamma, gammaMax, epsFwd, epsAwy, stepSize, delta;
        uint64_t fwdInd{}, awyInd{};
        Vector<V> fwdVec, awyVec;
        bool isFwd{};
    };
}// namespace mopmc::optimization::optimizers

#endif//MOPMC_FRANKWOLFE_H
