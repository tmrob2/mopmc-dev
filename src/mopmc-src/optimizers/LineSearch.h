//
// Created by guoxin on 26/11/23.
//

#ifndef MOPMC_LINESEARCH_H
#define MOPMC_LINESEARCH_H

#include <Eigen/Dense>

#include "../convex-functions/BaseConvexFunction.h"


namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class LineSearch {
    public:

        explicit LineSearch(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f_);

        V findOptimalPoint(Vector<V> &vLeft, Vector<V> &vRight);

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *f_;
        Vector<V> vLeft_, vRight_;
        V epsilon2 = 1e-4;

        V g(V lambda) {
            assert(lambda >= static_cast<V>(0.) && lambda <= static_cast<V>(1.));
            Vector<V> vOut = lambda * this->vLeft_ + (1-lambda) * this->vRight_;
            return f_->value(vOut);
        }

        V dg(V lambda) {
            Vector<V> v1 = lambda * this->vLeft_ + (1-lambda) * this->vRight_;
            Vector<V> v2 = this->vLeft_ - this->vRight_;
            return f_->subgradient(v1).dot(v2);
        }
    };
}


#endif //MOPMC_LINESEARCH_H
