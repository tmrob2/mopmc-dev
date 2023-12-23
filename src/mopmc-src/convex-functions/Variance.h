//
// Created by guoxin on 23/12/23.
//

#ifndef MOPMC_VARIANCE_H
#define MOPMC_VARIANCE_H

#include "BaseConvexFunction.h"
#include <Eigen/Dense>
#include <vector>

namespace mopmc::optimization::convex_functions {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class Variance : public BaseConvexFunction<V> {
    public:
        explicit Variance(uint64_t dim);
        V value(const Vector<V> &x) override;
        Vector<V> subgradient(const Vector<V> &x) override;
    };

    template<typename V>
class StandDeviation : public BaseConvexFunction<V> {
    public:
        explicit StandDeviation(uint64_t dim);
        V value(const Vector<V> &x) override;
        Vector<V> subgradient(const Vector<V> &x) override;
    };
}


#endif//MOPMC_VARIANCE_H
