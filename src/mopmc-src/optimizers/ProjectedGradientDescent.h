//
// Created by guoxin on 4/12/23.
//

#ifndef MOPMC_PROJECTEDGRADIENTDESCENT_H
#define MOPMC_PROJECTEDGRADIENTDESCENT_H

#include <vector>
#include <Eigen/Dense>
#include "../convex-functions/BaseConvexFunction.h"

namespace mopmc::optimization::optimizers{
    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    class ProjectedGradientDescent {
    public:

        explicit ProjectedGradientDescent(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

         Vector<V> findNearestProjectedPoint(Vector<V> &x,
                                             std::vector<Vector<V>> &Phi,
                                             std::vector<Vector<V>> &W);
         
         Vector<V> argmin(Vector<V> &iniPoint,
                          std::vector<Vector<V>> &Phi,
                          std::vector<Vector<V>> &W);

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;

    };
}


#endif //MOPMC_PROJECTEDGRADIENTDESCENT_H
