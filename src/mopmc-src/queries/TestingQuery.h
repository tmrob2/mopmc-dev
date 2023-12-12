//
// Created by guoxin on 27/11/23.
//

#ifndef MOPMC_TESTINGQUERY_H
#define MOPMC_TESTINGQUERY_H

#include <Eigen/Dense>
#include <vector>
#include "BaseQuery.h"
#include "../solvers/CudaValueIteration.cuh"

namespace mopmc::queries {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V, typename I>
    class TestingQuery : public BaseQuery<V, I>{
    public:

        explicit TestingQuery(const mopmc::QueryData<V,I> &data) : BaseQuery<V, I>(data) {};
        TestingQuery(const mopmc::QueryData<V,I> &data,
                     mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                     mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                     mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt)
                : BaseQuery<V, I>(data, f, priOpt, secOpt) {};
        void query() override;

    };

}


#endif //MOPMC_TESTINGQUERY_H
