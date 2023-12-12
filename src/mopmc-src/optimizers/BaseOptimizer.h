//
// Created by guoxin on 11/12/23.
//

#ifndef MOPMC_BASEOPTIMIZER_H
#define MOPMC_BASEOPTIMIZER_H

#include <vector>
#include <Eigen/Dense>
#include "../convex-functions/BaseConvexFunction.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class BaseOptimizer {
    public:
        explicit BaseOptimizer<V>() = default;
        explicit BaseOptimizer<V>(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f) : fn(f){};

        virtual int minimize (Vector<V> &point, const std::vector<Vector<V>> &Vertices) { return 1; }

        virtual int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices,
                     const std::vector<Vector<V>> &Weights) { return 1; }

        virtual int minimize () { return 1; }

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;
    };

}

#endif //MOPMC_BASEOPTIMIZER_H
