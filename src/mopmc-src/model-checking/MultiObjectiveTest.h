//
// Created by thomas on 20/08/23.
//

#ifndef MOPMC_MULTIOBJECTIVETEST_H
#define MOPMC_MULTIOBJECTIVETEST_H
#include <iostream>
#include <memory>
#include <storm/environment/Environment.h>
#include <storm/logic/MultiObjectiveFormula.h>
#include "mopmc-src/SparseModel.h"

namespace mopmc {
    namespace stormtest{
        template<typename SparseModelType>
        //std::unique_ptr<storm::modelchecker::CheckResult>
        void performMultiObjectiveModelChecking(
            storm::Environment env,
            SparseModelType const& model,
            storm::logic::MultiObjectiveFormula const& formula
            //mopmc::sparse::SparseModelBuilder<double>& spMat
            );
    }
}




#endif //MOPMC_MULTIOBJECTIVETEST_H
