//
// Created by guoxin on 4/12/23.
//

#ifndef MOPMC_GPUCONVEXQUERYALT_H
#define MOPMC_GPUCONVEXQUERYALT_H
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/api/storm.h>
#include "BaseQuery.h"
#include "../Data.h"

namespace mopmc::queries {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename T, typename I>
    class GpuConvexQueryAlt : public BaseQuery<T, I>{
    public:
        explicit GpuConvexQueryAlt(const mopmc::Data<T, I> &data) : BaseQuery<T, I>(data) {};

        void query() override;
    };
}

#endif //MOPMC_GPUCONVEXQUERYALT_H
