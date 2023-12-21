//
// Created by guoxin on 21/12/23.
//

#ifndef MOPMC_QUERYOPTIONS_H
#define MOPMC_QUERYOPTIONS_H

#include "convex-functions/BaseConvexFunction.h"
#include "optimizers/BaseOptimizer.h"
#include "solvers/BaseValueIteration.h"

namespace mopmc {

    struct QueryOptions {

        explicit QueryOptions() = default;

        enum {MSE, EUCLIDEAN} CONVEX_FUN;
        enum {BLENDED, BLENDED_STEP_OPT, AWAY_STEP, LINOPT} PRIMARY_OPTIMIZER, SECONDARY_OPTIMIZER;
        enum {CUDA_VI} VI;

    };
    /*
    template<typename V>
    struct QueryOptions {

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *convexFunction;
        mopmc::optimization::optimizers::BaseOptimizer<V> *primaryOptimizer;
        mopmc::optimization::optimizers::BaseOptimizer<V> *secondOptimizer;
        mopmc::value_iteration::BaseVIHandler<V> *viHandler;

        void initialize(const QueryOptionsEnums &queryOptionsEnums) {
            switch (queryOptionsEnums.CONVEX_FUN) {
                case QueryOptionsEnums::MSE:
                    convexFunction = ;
                case QueryOptionsEnums::EUCLIDEAN:
                    break;
            }
        }

    };

    template class QueryOptions<double>;
        */
}



#endif //MOPMC_QUERYOPTIONS_H
