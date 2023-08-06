//
// Created by thomas on 2/08/23.
//

#ifndef MOPMC_REWARDMODELBUILDER_H
#define MOPMC_REWARDMODELBUILDER_H

#include <storm/builder/RewardModelInformation.h>
#include <storm/models/sparse/StandardRewardModel.h>

namespace mopmc {
    template<typename ValueType>
    class RewardModelBuilder {
    public:
        RewardModelBuilder(storm::builder::RewardModelInformation const& rewardModelInformation)
        : 
            rewardModelName(rewardModelInformation.getName()),
            stateRewards(rewardModelInformation.hasStateRewards()),
            stateActionRewards(rewardModelInformation.hasStateActionRewards()),
            stateRewardVector(),
            stateActionRewardVector()
        {}

        storm::models::sparse::StandardRewardModel<ValueType> build(
            uint_fast64_t rowCount, uint_fast64_t columnCount, uint_fast64_t rowGroupCount);

        std::string const& getName() const;

        void addStateReward(ValueType const& value);

        void addStateActionReward(ValueType const& value);

        bool hasStateRewards() const;

        bool hasStateActionRewards() const;

        std::vector<ValueType>& getStateRewardVector() {
            return this->stateRewardVector;
        }

    private:
        std::string rewardModelName;

        bool stateRewards;
        bool stateActionRewards;

        // The state reward vector.
        std::vector<ValueType> stateRewardVector;

        // The state-action reward vector.
        std::vector<ValueType> stateActionRewardVector;
    };
}

#endif //MOPMC_REWARDMODELBUILDER_H
