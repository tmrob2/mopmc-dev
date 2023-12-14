//
// Created by guoxin on 13/12/23.
//

#ifndef MOPMC_KLDISTANCE_H
#define MOPMC_KLDISTANCE_H
#include "BaseConvexFunction.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    class KLDistance : public BaseConvexFunction<V> {
    public:
        explicit KLDistance(const Vector<V> &c);

        V value(const Vector<V> &x) override;

        Vector<V> subgradient(const Vector<V> &x) override;

        V probabilityLowerBound, probabilityUpperBound;
        V gradientLowerBound, gradientUpperBound;

    private:
        V klDivergence(V x, V c);
        V d_klDivergence(V x, V c);
    };
}


#endif //MOPMC_KLDISTANCE_H
