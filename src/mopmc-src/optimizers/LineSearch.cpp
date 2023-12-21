//
// Created by guoxin on 26/11/23.
//

#include "LineSearch.h"
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    LineSearcher<V>::LineSearcher(convex_functions::BaseConvexFunction<V> *f) : f_(f) {}

    template<typename V>
    V LineSearcher<V>::findOptimalDecentDistance(Vector<V> vLeft, Vector<V> vRight, V lambdaMax) {

        const V epsilon2 = 1e-12;
        this->vLeft_ = vLeft;
        this->vRight_ = vRight;
        V lambda0 = static_cast<V>(0.), lambda1 = lambdaMax;
        if (dg(lambda0) == static_cast<V>(0.)) {
            return lambda0;
        }
        if (dg(lambda1) == static_cast<V>(0.)) {
            return lambda1;
        }
        V delta = static_cast<V>(1.);
        uint64_t iter = 0;
        while (lambda1 - lambda0 > epsilon2 ) {
            if (dg(lambda0) * dg(static_cast<V>(0.5) * (lambda0 + lambda1)) > static_cast<V>(0.) ) {
                lambda0 = static_cast<V>(0.5) * (lambda0 + lambda1);
            } else if (dg(lambda0) * dg(static_cast<V>(0.5) * (lambda0 + lambda1)) < static_cast<V>(0.)) {
                lambda1 = static_cast<V>(0.5) * (lambda0 + lambda1);
            } else {
                break;
            }
            ++iter;
        }

        return static_cast<V>(0.5) * (lambda0 + lambda1);
    }

    template<typename V>
    V LineSearcher<V>::g(V lambda) {
        assert(lambda >= static_cast<V>(0.) && lambda <= static_cast<V>(1.));
        Vector<V> vOut = (static_cast<V>(1.) - lambda) * this->vLeft_ + lambda * this->vRight_;
        return this->f_->value(vOut);
    }

    template<typename V>
    V LineSearcher<V>::dg(V lambda) {
        Vector<V> v1 = (static_cast<V>(1.) - lambda) * this->vLeft_ + lambda * this->vRight_;
        Vector<V> v2 = this->vRight_ - this->vLeft_;
        return this->f_->subgradient(v1).dot(v2);
    }

    template class LineSearcher<double>;

}
