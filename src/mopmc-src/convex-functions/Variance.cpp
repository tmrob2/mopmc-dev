//
// Created by guoxin on 23/12/23.
//

#include "Variance.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    Variance<V>::Variance(const uint64_t dim) : BaseConvexFunction<V>(dim){
        this->smooth = true;
    }

    template<typename V>
    V Variance<V>::value(const Vector<V> &x) {
        V e = x.mean();
        V y = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += std::pow(x(i) - e, 2);
        }
        return y;
    }

    template<typename V>
    Vector<V> Variance<V>::subgradient(const Vector<V> &x) {
        V e = x.mean();
        uint64_t d = x.size();
        Vector<V> y(x.size());
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y(i) = 2 * (x(i) - e) * (d-1) / d;
        }
        return y;
    }

    template<typename V>
    StandDeviation<V>::StandDeviation(uint64_t dim) : BaseConvexFunction<V>(dim) {
        this->smooth = true;
    }

    template<typename V>
    V StandDeviation<V>::value(const Vector<V> &x) {
        V e = x.mean();
        V y = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += std::pow(x(i) - e, 2);
        }
        return std::sqrt(y);
    }

    template<typename V>
    Vector<V> StandDeviation<V>::subgradient(const Vector<V> &x) {
        V e = x.mean();
        uint64_t d = x.size();
        V g = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            g += std::pow(x(i) - e, 2);
        }
        g = std::sqrt(g);
        Vector<V> y(x.size());
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y(i) = ((x(i) - e) * (d-1) / d) / g;
        }
        return y;
    }

    template class Variance<double>;
    template class StandDeviation<double>;
}
