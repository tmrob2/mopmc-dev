//
// Created by guoxin on 20/11/23.
//

#include "Transformation.h"
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm/storage/prism/Program.h>
#include <string>
#include <iostream>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/environment/Environment.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include "model-checking/MOPMCModelChecking.h"

namespace mopmc {
    template<typename M, typename V, typename I>
     mopmc::Data<V, I> Transformation<M, V, I>::transform(
            typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &prepReturn) {

        mopmc::Data<V, I> data;
        mopmc::multiobjective::MOPMCModelChecking<M> model(prepReturn);

        data.transitionMatrix = *storm::adapters::EigenAdapter::toEigenSparseMatrix(model.getTransitionMatrix());
        data.transitionMatrix.makeCompressed();
        data.rowCount = model.getTransitionMatrix().getRowCount();
        data.colCount = model.getTransitionMatrix().getColumnCount();
        data.rowGroupIndices = model.getTransitionMatrix().getRowGroupIndices();
        data.row2RowGroupMapping.resize(data.rowCount);

        for (uint64_t i = 0; i < data.rowGroupIndices.size() - 1; ++i) {
            size_t currInd = data.rowGroupIndices[i];
            size_t nextInd = data.rowGroupIndices[i + 1];
            for (uint64_t j = 0; j < nextInd - currInd; ++j)
                data.row2RowGroupMapping[currInd + j] = i;
        }
        data.objectiveCount = prepReturn.objectives.size();
        data.rewardVectors = model.getActionRewards();
        assert(data.rewardVectors.size() == data.objectiveCount);
        assert(data.rewardVectors[0].size() == data.rowCount);
        data.flattenRewardVector.resize(data.objectiveCount * data.rowCount);
        for (uint64_t i = 0; i < data.objectiveCount; ++i) {
            for (uint_fast64_t j = 0; j < data.rowCount; ++j) {
                data.flattenRewardVector[i * data.rowCount + j] = data.rewardVectors[i][j];
            }
        }
        data.thresholds.resize(data.objectiveCount);
        data.probObjectives.resize(data.objectiveCount);
        for (uint_fast64_t i = 0; i < data.objectiveCount; ++i) {
            data.thresholds[i] = prepReturn.objectives[i].formula->template getThresholdAs<V>();
            data.probObjectives[i] = prepReturn.objectives[i].originalFormula->isProbabilityOperatorFormula();
        }

        data.defaultScheduler.assign(data.colCount, static_cast<uint64_t>(0));
        data.initialRow = model.getInitialState();

        return data;
    }

    template class mopmc::Transformation<storm::models::sparse::Mdp<double>, double, uint64_t>;

}
