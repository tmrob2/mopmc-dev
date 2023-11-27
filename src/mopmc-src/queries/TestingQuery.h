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

    template<typename T>
    class TestingQuery : public BaseQuery<T>{
    public:

        explicit TestingQuery(const mopmc::Data<T,uint64_t> &data) : BaseQuery<T>(data) {};
        void query() override;

    };

}


#endif //MOPMC_TESTINGQUERY_H
