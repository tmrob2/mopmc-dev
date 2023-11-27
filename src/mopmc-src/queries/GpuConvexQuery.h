//
// Created by guoxin on 2/11/23.
//
#ifndef MOPMC_CONVEXXQUERY_H
#define MOPMC_CONVEXXQUERY_H
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

    template<typename T>
    class GpuConvexQuery : public BaseQuery<T>{
    public:
        explicit GpuConvexQuery(const mopmc::Data<T, uint64_t> &data) : BaseQuery<T>(data) {};

        void query() override;

        Eigen::SparseMatrix<T, Eigen::RowMajor> P_;
        //storm::Environment env_;
        //mopmc::Data<T, uint64_t> data_;
    };
}

#endif //MOPMC_CONVEXXQUERY_H
