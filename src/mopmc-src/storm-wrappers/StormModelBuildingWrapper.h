//
// Created by guoxin on 17/11/23.
//

#ifndef MOPMC_STORMMODELBUILDINGWRAPPER_H
#define MOPMC_STORMMODELBUILDINGWRAPPER_H

#include <string>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <storm/adapters/EigenAdapter.h>
#include <storm/environment/Environment.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>


namespace  mopmc {

    template<typename M>
    class ModelBuilder : public storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<M> {
    public:
        explicit ModelBuilder(typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType returnType) :
                storm::modelchecker::multiobjective::StandardMdpPcaaWeightVectorChecker<M>(returnType){};

        static typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType preprocess(
                const std::string &path_to_model, const std::string &property_string, storm::Environment &env);

        static ModelBuilder<M> build(
                typename storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<M>::ReturnType &preliminaryData);

        storm::storage::SparseMatrix<typename M::ValueType> getTransitionMatrix() {
            return this->transitionMatrix;
        }

        std::vector<std::vector<typename M::ValueType>> getActionRewards() {
            return this->actionRewards;
        }

        [[nodiscard]] uint64_t getInitialState() const {
            return this->initialState;
        }

    };

}

#endif //MOPMC_STORMMODELBUILDINGWRAPPER_H
