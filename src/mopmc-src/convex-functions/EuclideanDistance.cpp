//
// Created by guoxin on 4/12/23.
//

#include "EuclideanDistance.h"
#include <algorithm>
#include <cmath>


namespace mopmc::optimization::convex_functions {

    template<typename V>
    EuclideanDistance<V>::EuclideanDistance(const Vector<V> &c):BaseConvexFunction<V>(c) {
        this->smooth = true;
    }

    template<typename V>
    Vector<V> EuclideanDistance<V>::subgradient(const Vector<V> &x) {
        Vector<V> y(x.size());
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y(i) = 2 * (x(i) - this->parameters(i));
        }
        return y;
    }

    template<typename V>
    V EuclideanDistance<V>::value(const Vector<V> &x) {
        V y = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += std::pow(x(i) - this->parameters(i), 2);
        }
        return y;
    }

    template class EuclideanDistance<double>;
}