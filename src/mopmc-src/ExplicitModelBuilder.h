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
#include <storm/builder/StateAndChoiceInformationBuilder.h>
#include <storm/builder/RewardModelBuilder.h>

#include <deque>


namespace mopmc {
    bool check(std::string const& path_to_model, std::string const& property_string);

    bool stormCheck(std::string const& path_to_model, std::string const& property_string);

    bool stormWarehouseExperiment(std::string const& path_to_explicit_model, std::string const& path_to_explicit_labelling,
                                   std::vector<std::string> const& path_to_rewards_model, std::string const& propFname);

    std::string readPropFile(std::string const& fname);
}

#endif //MOPMC_EXPLICITMODELBUILDER_H
