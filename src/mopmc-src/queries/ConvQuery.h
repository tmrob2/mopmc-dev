//
// Created by guoxin on 2/11/23.
//
#ifndef MOPMC_CONVEXXQUERY_H
#define MOPMC_CONVEXXQUERY_H
#include "../Runner.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/api/storm.h>
#include "../Preprocessor.h"

namespace mopmc::queries {

    class ConvexQuery{
        typedef typename ModelType::ValueType T;

    public:

        ConvexQuery(const mopmc::PreprocessedData<ModelType> &data, const storm::Environment& env);

        void query();

        Eigen::SparseMatrix<T, Eigen::RowMajor> P_;
        storm::Environment env_;
        mopmc::PreprocessedData<ModelType> data_;
    };
}

#endif //MOPMC_CONVEXXQUERY_H
