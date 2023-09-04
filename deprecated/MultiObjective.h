//
// Created by thomas on 18/08/23.
//

#ifndef MOPMC_MULTIOBJECTIVE_H
#define MOPMC_MULTIOBJECTIVE_H

#include "SparseModel.h"
#include <storm/builder/RewardModelBuilder.h>
#include <storm/logic/MultiObjectiveFormula.h>
#include <boost/optional.hpp>
#include <string>

namespace mopmc{
namespace multiobj{
    template <typename ValueType>
    void performMultiObjectiveModelChecking(mopmc::sparse::SparseModelBuilder<ValueType>& spModel,
                                            storm::logic::MultiObjectiveFormula const& formula);

    template <typename ValueType>
    void preprocess(mopmc::sparse::SparseModelBuilder<ValueType>& spModel,
                    storm::logic::MultiObjectiveFormula const& formula);

    template <typename ValueType>
    void reduceStates(mopmc::sparse::SparseModelBuilder<ValueType>& spModel,
                      boost::optional<std::string> deadlockLabel,
                      storm::logic::MultiObjectiveFormula const& formula);

    template <typename ValueType>
    storm::storage::BitVector getOnlyReachableViaPhi(mopmc::sparse::SparseModelBuilder<ValueType>& spModel,
                                                     storm::storage::BitVector const& phi);

    template <typename ValueType>
    storm::storage::BitVector getReachableSubSystem(
        sparse::SparseModelBuilder<ValueType> &spModel,
        const storm::storage::BitVector &subsystemStates,
        const storm::storage::BitVector &subsystemActions);

    template <typename ValueType>
    void makeSubSystem(
        sparse::SparseModelBuilder<ValueType> &spModel,
        const storm::storage::BitVector &subsystemStates,
        const storm::storage::BitVector &subSystemActions
    );
}
}


#endif //MOPMC_MULTIOBJECTIVE_H
