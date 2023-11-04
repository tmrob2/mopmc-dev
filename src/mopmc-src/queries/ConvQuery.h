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
        ConvexQuery(const PrepReturnType& t_);
        ConvexQuery(const PrepReturnType& t_, const storm::Environment& env_);

        void query();
        PrepReturnType t;
        storm::Environment env;
    };
    //TODO

}

#endif //MOPMC_CONVEXXQUERY_H
