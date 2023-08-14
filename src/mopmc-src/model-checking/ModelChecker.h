//
// Created by thomas on 9/08/23.
//
#ifndef MOPMC_MODELCHECKER_H
#define MOPMC_MODELCHECKER_H

#include <storm/modelchecker/CheckTask.h>
#include <storm/logic/Formulas.h>
#include <storm/logic/ProbabilityOperatorFormula.h>
#include <storm/modelchecker/results/CheckResult.h>
#include <storm/environment/Environment.h>
#include <storm/solver/SolveGoal.h>
#include "../SparseModel.h"

#include <memory>

/* A stripped down version of the storm abstract model checker
 *
 * No fancy virtual function inheritence we are only doing probabilistic model
 * checking on MDP which has collapses to a DTMC under an optimal scheduler.
 * Therefore, we only need a DTMC and MDP model checker.
*/

namespace mopmc{
namespace model_checking{
template <typename ValueType>
class DTMCModelSolver {
public:
    DTMCModelSolver(mopmc::sparse::SparseModelBuilder<ValueType>& spMatBuilder);

    std::unique_ptr<storm::modelchecker::CheckResult> check(
            storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& checkTask);

    std::unique_ptr<storm::modelchecker::CheckResult> check(
            storm::Environment const& env,
            storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& checkTask);

    mopmc::sparse::SparseModelBuilder<ValueType>& getSparseMatrixBuilder();

    bool canHandle(storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const &checkTask);

    std::unique_ptr<storm::modelchecker::CheckResult> checkStateFormula(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::StateFormula, ValueType> const &checkTask
    );

    void checkProbabilityOperatorFormula(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::ProbabilityOperatorFormula, ValueType> const& checkTask);

    std::unique_ptr<storm::modelchecker::CheckResult> computeProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& checkTask);

    std::unique_ptr<storm::modelchecker::CheckResult> computeLTLProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& checkTask);

    std::unique_ptr<storm::modelchecker::CheckResult> computeReachabilityProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask);
        
    void computeUntilProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::UntilFormula, ValueType> const& checkTask);

    void computeUntilProbabilitiesHelper(
        storm::Environment const& env,
        storm::storage::BitVector const& relevantValues,
        mopmc::sparse::SparseModelBuilder<ValueType>& spModel,
        storm::storage::BitVector const& phiStates,
        storm::storage::BitVector const& psiStates,
        storm::modelchecker::ModelCheckerHint const& hint);

    void computeNextProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask);

    std::unique_ptr<storm::modelchecker::CheckResult> checkBooleanLiteralFormula(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::BooleanLiteralFormula, ValueType> const& checkTask
    );

    std::unique_ptr<storm::modelchecker::CheckResult> checkAtomicLabelFormula(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::AtomicLabelFormula, ValueType> const& checkTask
    );


private:
    /*!
     * The sparse matrix builder has access to transition matrices and reward matrices without
     * taking ownership of this memory
     */
    mopmc::sparse::SparseModelBuilder<ValueType>& spModel;
};
}
}

#endif //MOPMC_MODELCHECKER_H