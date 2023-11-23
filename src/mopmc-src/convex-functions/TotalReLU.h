//
// Created by guoxin on 21/11/23.
//

#ifndef MOPMC_TOTALRELU_H
#define MOPMC_TOTALRELU_H

#include <vector>
#include "BaseConvexFunction.h"

namespace mopmc::optimisation::convex_functions {

    template<typename V>
    class TotalReLU : public BaseConvexFunction<V> {
    public:

        explicit TotalReLU(std::vector<V> &c);

        V value(std::vector<V> &x) override;

        std::vector<V> subgradient(std::vector<V> &x) override;

        std::vector<V> c_;
    };
}

#endif //MOPMC_TOTALRELU_H
