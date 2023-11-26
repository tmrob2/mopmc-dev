//
// Created by guoxin on 2/11/23.
//
#ifndef MOPMC_CONVEXXQUERY_H
#define MOPMC_CONVEXXQUERY_H
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/api/storm.h>
#include "../Data.h"

namespace mopmc::queries {

    template<typename T>
    class ConvexQuery{

    public:

        //ConvexQuery(const mopmc::Data<T, uint64_t> &data, const storm::Environment& env);
        explicit ConvexQuery(const mopmc::Data<T, uint64_t> &data);

        void query();

        Eigen::SparseMatrix<T, Eigen::RowMajor> P_;
        storm::Environment env_;
        mopmc::Data<T, uint64_t> data_;
    };
}

#endif //MOPMC_CONVEXXQUERY_H
