//
// Created by guoxin on 10/12/23.
//

#ifndef MOPMC_LINCOM_H
#define MOPMC_LINCOM_H

#include <Eigen/Dense>

namespace mopmc::optimization::convex_functions::auxiliary {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class LinearCombination {
    public:
        explicit LinearCombination() = default;

    };
}


#endif //MOPMC_LINCOM_H
