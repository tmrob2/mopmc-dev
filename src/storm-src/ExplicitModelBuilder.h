//
// Created by thomas on 2/08/23.
//

#ifndef MOPMC_EXPLICITMODELBUILDER_H
#define MOPMC_EXPLICITMODELBUILDER_H

#include <storm/generator/NextStateGenerator.h>
#include <storm/builder/ExplorationOrder.h>
#include <storm/settings/modules/BuildSettings.h>
#include <storm/settings/SettingsManager.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/storage/sparse/StateStorage.h>
#include "TransitionMatrixBuilder.h"

#include <deque>


namespace mopmc {
    typedef storm::storage::BitVector CompressedState;
    
    template<typename ValueType, typename RewardModelType=storm::models::sparse::StandardRewardModel<ValueType>, typename StateType=uint32_t>
    class ExplicitModelBuilder {
    public:
        struct Options {
            Options() : explorationOrder(storm::settings::getModule<storm::settings::modules::BuildSettings>().getExplorationOrder()) {
                // Intentionally left blank
            };
            storm::builder::ExplorationOrder explorationOrder;
        };

        ExplicitModelBuilder(std::shared_ptr<storm::generator::NextStateGenerator<double, uint32_t>> const& generator,
                             Options const& options = Options()) : 
                             generator {generator }, options { options }, stateStorage {generator -> getStateSize() } {}

        void buildMatrices(
            SparseMatrixBuilder& transitionMatrixBuilder,
            std::vector<storm::builder::RewardModelBuilder<typename RewardModelType::ValueType>>& rewardModelBuilders,
            storm::builder::StateAndChoiceInformationBuilder& stateAndChoiceInformationBuilder
        );

        storm::storage::sparse::StateStorage<StateType>& getStateStorage();
    private:
        std::shared_ptr<storm::generator::NextStateGenerator<double, uint32_t>> generator;
        Options options;

        /*!
        * Retrieves the state id of the given state. If the state has not been encountered yet, it will be added to
        * the lists of all states with a new id. If the state was already known, the object that is pointed to by
        * the given state pointer is deleted and the old state id is returned. Note that the pointer should not be
        * used after invoking this method.
        *
        * @param state A pointer to a state for which to retrieve the index. This must not be used after the call.
        * @return A pair indicating whether the state was already discovered before and the state id of the state.
        */
        StateType getOrAddStateIndex(CompressedState const& state);

        /// Internal information about the states that were explored.
        storm::storage::sparse::StateStorage<StateType> stateStorage;

        std::deque<std::pair<CompressedState, StateType>> statesToExplore;
    };

    bool check(std::string const& path_to_model, std::string const& property_string);
}

#endif //MOPMC_EXPLICITMODELBUILDER_H
