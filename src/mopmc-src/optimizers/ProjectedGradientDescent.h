//
// Created by guoxin on 4/12/23.
//

#ifndef MOPMC_PROJECTEDGRADIENTDESCENT_H
#define MOPMC_PROJECTEDGRADIENTDESCENT_H

#include <vector>
#include <Eigen/Dense>
#include <memory>
#include "../convex-functions/BaseConvexFunction.h"

namespace mopmc::optimization::optimizers{

    enum ProjectionType {
        NearestHyperplane, UnitSimplex
    };

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class ProjectedGradientDescent {
    public:

        explicit ProjectedGradientDescent(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        ProjectedGradientDescent(ProjectionType type, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        Vector<V> projectToNearestHyperplane(Vector<V> &x,
                                              std::vector<Vector<V>> &Phi,
                                              std::vector<Vector<V>> &W);

        Vector<V> projectToUnitSimplex(Vector<V> &x);

        Vector<V> argminNearestHyperplane(Vector<V> &iniPoint,
                                          std::vector<Vector<V>> &Phi,
                                          std::vector<Vector<V>> &W);


        Vector<V> argminUnitSimplexProjection(Vector<V> &weightVector,
                                               std::vector<Vector<V>> &Points);

        Vector<V> argmin(std::vector<Vector<V>> &Vertices,
                         std::vector<Vector<V>> &Weights,
                         Vector<V> &initialPoint);

        Vector<V> argmin(std::vector<Vector<V>> &Vertices,
                         Vector<V> &initialPoint);

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;
        ProjectionType projectionType{};


    };
}


#endif //MOPMC_PROJECTEDGRADIENTDESCENT_H
