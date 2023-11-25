//
// Created by guoxin on 21/11/23.
//

#ifndef MOPMC_TOTALRELU_H
#define MOPMC_TOTALRELU_H

#include <vector>
#include <Eigen/Dense>
#include "BaseConvexFunction.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class TotalReLU : public BaseConvexFunction<V> {
    public:

        explicit TotalReLU(std::vector<V> &c);
        explicit TotalReLU(Vector<V> &e);
        explicit TotalReLU(VectorMap<V> &e);

        V value(std::vector<V> &x) override;
        V value1(Vector<V> &x) override;

        std::vector<V> subgradient(std::vector<V> &x) override;
        Vector<V> subgradient1(Vector<V> &x) override;

        std::vector<V> c_;
    };


}

#endif //MOPMC_TOTALRELU_H
