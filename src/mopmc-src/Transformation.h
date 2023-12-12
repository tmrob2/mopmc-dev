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
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include "QueryData.h"
#include "mopmc-src/storm-wrappers/StormModelBuildingWrapper.h"

namespace mopmc {
    template<typename M, typename V, typename I>
    class Transformation {
    public:

        static std::shared_ptr<QueryData<V, int>> transform_i32(
                typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn,
                mopmc::ModelBuilder<M> &model);

        static mopmc::QueryData<V, int> transform_i32_v2(
                typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn,
                mopmc::ModelBuilder<M> &prepModel);

    };

}


#endif //MOPMC_TRANSFORMATION_H
