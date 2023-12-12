//
// Created by guoxin on 27/11/23.
//

#ifndef MOPMC_BASEQUERY_H
#define MOPMC_BASEQUERY_H

#include <storm/api/storm.h>
#include "../QueryData.h"
#include "../convex-functions/BaseConvexFunction.h"
#include "../optimizers/BaseOptimizer.h"
#include "../solvers/BaseValueIteration.h"

namespace mopmc::queries {

    template<typename V, typename I>
    class BaseQuery {
    public:

        explicit BaseQuery() = default;
        explicit BaseQuery(const mopmc::QueryData<V,I> &data): data_(data){};
        explicit BaseQuery(const mopmc::QueryData<V,I> &data,
                           mopmc::value_iteration::BaseVIHandler<V> *valueIterSolver)
                           : data_(data), VIhandler(valueIterSolver){};
        explicit BaseQuery(const mopmc::QueryData<V,I> &data,
                           mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt):
                           data_(data), fn(f), primaryOptimizer(priOpt), secondaryOptimizer(secOpt){};
        explicit BaseQuery(const mopmc::QueryData<V,I> &data,
                           mopmc::optimization::convex_functions::BaseConvexFunction<V> *f,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *priOpt,
                           mopmc::optimization::optimizers::BaseOptimizer<V> *secOpt,
                           mopmc::value_iteration::BaseVIHandler<V> *valueIterSolver):
                data_(data), fn(f), primaryOptimizer(priOpt), secondaryOptimizer(secOpt), VIhandler(valueIterSolver){};

        virtual void query() = 0 ;

        mopmc::optimization::convex_functions::BaseConvexFunction<V> *fn;
        mopmc::optimization::optimizers::BaseOptimizer<V> *primaryOptimizer;
        mopmc::optimization::optimizers::BaseOptimizer<V> *secondaryOptimizer;
        mopmc::value_iteration::BaseVIHandler<V> *VIhandler;
        mopmc::QueryData<V, I> data_;
    };



}

#endif //MOPMC_BASEQUERY_H
