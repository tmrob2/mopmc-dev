//
// Created by guoxin on 2/11/23.
//
// @_@ Insert a typo to differentiate it from an existing one
#ifndef MOPMC_CONVEXXQUERY_H
#define MOPMC_CONVEXXQUERY_H
#include "../Runner.h"
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>

namespace mopmc::queries {

    class ConvexQuery{
    public:
        //ConvexQuery() = default;
        ConvexQuery(const PrepReturnType& rt,
                    const EigenSpMatrix& eg
                    );
        void query();
        PrepReturnType t;
        EigenSpMatrix e;
        int numObjs;


    };
    //TODO

}

#endif //MOPMC_CONVEXXQUERY_H
