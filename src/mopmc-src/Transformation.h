//
// Created by guoxin on 20/11/23.
//

#ifndef MOPMC_TRANSFORMATION_H
#define MOPMC_TRANSFORMATION_H

#include <string>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/environment/Environment.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include "Data.h"

namespace mopmc {
    template<typename M, typename V, typename I>
    class Transformation {
    public:
        static mopmc::Data<V, I> transform(
                typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn);

        static mopmc::Data<V, int> transform_i32(
                typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn);

    };


}


#endif //MOPMC_TRANSFORMATION_H
