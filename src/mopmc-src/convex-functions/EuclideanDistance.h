//
// Created by guoxin on 4/12/23.
//

#ifndef MOPMC_EUCLIDEANDISTANCE_H
#define MOPMC_EUCLIDEANDISTANCE_H

#include <vector>
#include <Eigen/Dense>
#include "BaseConvexFunction.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class EuclideanDistance : public BaseConvexFunction<V>{
    public:
        explicit EuclideanDistance(Vector<V> &c);

        V value1(std::vector<V> &x) override;
        V value(Vector<V> &x) override;

        std::vector<V> subgradient1(std::vector<V> &x) override;
        Vector<V> subgradient(Vector<V> &x) override;

    };

}


#endif //MOPMC_EUCLIDEANDISTANCE_H
