//
// Created by guoxin on 18/12/23.
//

#include "MSE.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    MSE<V>::MSE(const Vector<V> &c, const uint64_t &n) :  numObjs(n), BaseConvexFunction<V>(c){
        this->smooth = true;
    }

    template<typename V>
    Vector<V> MSE<V>::subgradient(const Vector<V> &x) {
        Vector<V> y(x.size());
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y(i) = 2 * (x(i) - this->params_(i));
        }
        return y / this->numObjs;
    }

    template<typename V>
    V MSE<V>::value(const Vector<V> &x) {
        V y = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += std::pow(x(i) - this->params_(i), 2);
        }
        return y / this->numObjs;
    }

    template class MSE<double>;
}