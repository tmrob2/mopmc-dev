//
// Created by guoxin on 18/12/23.
//

#ifndef MOPMC_MSE_H
#define MOPMC_MSE_H

#include <vector>
#include <Eigen/Dense>
#include "BaseConvexFunction.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class MSE : public BaseConvexFunction<V> {
    public:
        explicit MSE(const Vector<V> &c, const uint64_t &n);

        V value(const Vector<V> &x) override;

        Vector<V> subgradient(const Vector<V> &x) override;

        uint64_t numObjs{};

    };
}


#endif //MOPMC_MSE_H
