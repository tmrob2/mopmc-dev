//
// Created by guoxin on 26/11/23.
//

#include "LineSearch.h"
namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    LineSearch<V>::LineSearch(convex_functions::BaseConvexFunction<V> *f) : f_(f) {}

    template<typename V>
    V LineSearch<V>::findOptimalPoint(Vector<V> &vLeft, Vector<V> &vRight) {
        this->vLeft_ = vLeft;
        this->vRight_ = vRight;
        V lambda1 = static_cast<V>(1.), lambda2 = static_cast<V>(0.);
        if (dg(lambda1) * dg(lambda2) >= lambda2) {
            if (g(lambda1) >= g(lambda2)) {
                return lambda1;
            } else {
                return lambda2;
            }
        }
        V delta = static_cast<V>(1.);
        while (delta > epsilon2 ) {
            if ( dg(lambda1) * dg(static_cast<V>(0.5) * (lambda1+lambda2)) >= static_cast<V>(0.) ) {
                lambda1 = static_cast<V>(0.5) * (lambda1+lambda2);
            } else {
                lambda2 = static_cast<V>(0.5) * (lambda1+lambda2);
            }
            delta = static_cast<V>(0.5) * delta;
        }
        return static_cast<V>(0.5) * (lambda1+lambda2);
    }

    template class LineSearch<double>;

}
