//
// Created by guoxin on 4/12/23.
//

#ifndef MOPMC_PROJECTEDGRADIENTDESCENT_H
#define MOPMC_PROJECTEDGRADIENTDESCENT_H

#include <vector>
#include <Eigen/Dense>
#include <memory>
#include "../convex-functions/BaseConvexFunction.h"
#include "BaseOptimizer.h"

namespace mopmc::optimization::optimizers{

    enum ProjectionType {
        NearestHyperplane, UnitSimplex
    };

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class ProjectedGradientDescent : public BaseOptimizer<V>{
    public:

        explicit ProjectedGradientDescent(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);

        ProjectedGradientDescent(ProjectionType type, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f);


        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) override;

        int minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices,
                     const std::vector<Vector<V>> &Weights) override;

        ProjectionType projectionType{};

        Vector<V> point_;

    private:
        Vector<V> argminUnitSimplexProjection(Vector<V> &weightVector,
                                              const std::vector<Vector<V>> &Points);

        Vector<V> argminNearestHyperplane(Vector<V> &iniPoint,
                                          const std::vector<Vector<V>> &Phi,
                                          const std::vector<Vector<V>> &W);

        Vector<V> projectToNearestHyperplane(Vector<V> &x,
                                             const std::vector<Vector<V>> &Phi,
                                             const std::vector<Vector<V>> &W);

        Vector<V> projectToUnitSimplex(Vector<V> &x);

        Vector<V> alpha;

    };
}


#endif //MOPMC_PROJECTEDGRADIENTDESCENT_H
