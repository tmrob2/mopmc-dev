//
// Created by thomas on 2/08/23.
//
#include <optional>

#include "RewardModelBuilder.h"

template<typename ValueType>
StandardRewardModel<ValueType> RewardModelBuilder<ValueType>::build(uint_fast64_t rowCount, uint_fast64_t columnCount,
                                                                    uint_fast64_t rowGroupCount) {
    std::optional<std::vector<ValueType>> optionalStateRewardVector;

    if (hasStateRewards()) {
        stateRewardVector.resize(rowGroupCount);
        optionalStateRewardVector = std::move(stateRewardVector);
    }

    std::optional<std::vector<ValueType>> optionalStateActionRewardVector;
    if (hasStateActionRewards()) {
        stateActionRewardVector.resize(rowCount);
        optionalStateActionRewardVector = std::move(stateActionRewardVector);
    }

    return StandardRewardModel<ValueType>(std::move(optionalStateRewardVector), std::move(optionalStateActionRewardVector));
}