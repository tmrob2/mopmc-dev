//
// Created by guoxin on 10/12/23.
//

#include <iostream>
#include "Lincom.h"

namespace mopmc::optimization::convex_functions::auxiliary {

    template<typename V>
    LinearCombination<V>::LinearCombination(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                                            const std::vector<Vector<V>> &Points) :
            fn(f), P(Points){}

    template<typename V>
    Vector<V> LinearCombination<V>::gradient(Vector<V> &x) {
        assert(this->fn->smooth == true);
        assert(!this->P.empty());
        assert(x.size() == this->P.size());
        uint64_t m = this->P[0].size();
        Vector<V> y(m);
        y.setZero();
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += (x(i) * this->P[i]);
        }
        Vector<V> k = this->fn->gradient(y);
        Vector<V> z(x.size());

        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            z(i) = k.dot(this->P[i]);
        }
        return z;
    }

    template<typename V>
    V LinearCombination<V>::value(Vector<V> &x) {
        uint64_t m = this->P[0].size();
        Vector<V> y(m);
        y.setZero();
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += (x(i) * this->P[i]);
        }
        return this->fn->value(y);
    }

    template class LinearCombination<double>;
}