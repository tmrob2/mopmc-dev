//
// Created by guoxin on 10/12/23.
//

#include "BaseConvexFunction.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    Vector<V> BaseConvexFunction<V>::gradient(const Vector<V> &x) {
        if (!smooth) {
            throw std::runtime_error("non smooth function does not have gradient, use subgradient");
        }
        return this->subgradient(x);
    }

    template<typename V>
    V BaseConvexFunction<V>::value1(const std::vector<V> &x) {
        std::vector<V> z = x;
        Vector<V> z1 = VectorMap<V>(z.data(), z.size());
        return this->value(z1);
    }

    template<typename V>
    std::vector<V> BaseConvexFunction<V>::subgradient1(const std::vector<V> &x) {
        std::vector<V> z = x;
        Vector<V> z1 = VectorMap<V>(z.data(), z.size());
        Vector<V> z2 = this->subgradient(z1);
        return std::vector<V>(z2.data(), z2.data() + z2.size());
    }

    template<typename V>
    std::vector<V> BaseConvexFunction<V>::gradient1(const std::vector<V> &x) {
        if (!smooth) {
            throw std::runtime_error("non smooth function does not have gradient");
        }
        return this->subgradient1(x);
    }

    template class BaseConvexFunction<double>;
}