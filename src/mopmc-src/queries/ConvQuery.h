//
// Created by guoxin on 2/11/23.
//
#ifndef MOPMC_CONVEXXQUERY_H
#define MOPMC_CONVEXXQUERY_H
#include "../Runner.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/api/storm.h>

namespace mopmc::queries {

    class ConvexQuery{
        typedef typename ModelType::ValueType T;

    public:
        explicit ConvexQuery(const PrepReturnType& model);
        ConvexQuery(const PrepReturnType& model, const storm::Environment& env);

        void query();
        PrepReturnType model_;
        storm::Environment env_;
    };
    //TODO

}

#endif //MOPMC_CONVEXXQUERY_H
