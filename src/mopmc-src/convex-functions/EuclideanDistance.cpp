//
// Created by guoxin on 4/12/23.
//

#include "EuclideanDistance.h"
#include <algorithm>
#include <cmath>


namespace mopmc::optimization::convex_functions {

    template<typename V>
    EuclideanDistance<V>::EuclideanDistance(Vector<V> &c):BaseConvexFunction<V>(c) {

    }

    template<typename V>
    Vector<V> EuclideanDistance<V>::subgradient(Vector<V> &x) {
        Vector<V> y(x.size());
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y(i) = 2 * (x(i) - this->e_(i));
        }
        return y;
    }

    template<typename V>
    std::vector<V> EuclideanDistance<V>::subgradient1(std::vector<V> &x) {
        return std::vector<V>();
    }

    template<typename V>
    V EuclideanDistance<V>::value(Vector<V> &x) {
        V y = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += std::pow(x(i) - this->e_(i), 2);
        }
        return y;
    }

    template<typename V>
    V EuclideanDistance<V>::value1(std::vector<V> &x) {
        return V();
    }

    template class EuclideanDistance<double>;
}