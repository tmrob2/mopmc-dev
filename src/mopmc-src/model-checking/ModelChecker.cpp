//
// Created by thomas on 9/08/23.
//
#include "ModelChecker.h"
#include "ModelCheckingUtils.h"
#include <storm/modelchecker/CheckTask.h>
#include <storm/logic/Formulas.h>
#include <storm/modelchecker/results/CheckResult.h>
#include <storm/environment/Environment.h>
#include <storm/logic/FormulaInformation.h>
#include <storm/logic/BooleanLiteralFormula.h>
#include <storm/modelchecker/results/ExplicitQualitativeCheckResult.h>
#include <storm/solver/LinearEquationSolver.h>
#include <storm/modelchecker/hints/ModelCheckerHint.h>
#include <storm/modelchecker/hints/ExplicitModelCheckerHint.h>
#include <storm/utility/vector.h>
#include <storm/solver/SolveGoal.h>

#include <memory>
#include <stdexcept>

namespace mopmc {
namespace model_checking{

    template<typename ValueType>
    DTMCModelSolver<ValueType>::DTMCModelSolver(mopmc::sparse::SparseModelBuilder<ValueType>& spModel)
    : spModel(spModel) {;
    }

    template<typename ValueType>
    bool DTMCModelSolver<ValueType>::canHandle(
            const storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> &checkTask) {
        return true;
    }

    template<typename ValueType>
    mopmc::sparse::SparseModelBuilder<ValueType>& DTMCModelSolver<ValueType>::getSparseMatrixBuilder() {
        return this -> spModel;
    }

    template <typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::check(
            storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& checkTask) {
        storm::Environment env;
        this -> check(env, checkTask);
    }

    template <typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::check(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& checkTask
    ) {
        storm::logic::Formula const& formula = checkTask.getFormula();
        if (formula.isStateFormula()) {
            std::cout << "A new state formula\n";
            return this->checkStateFormula(env, checkTask.substituteFormula(formula.asStateFormula()));
        }
        throw std::runtime_error("check: Not implemented error");
    }

    template<typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::checkStateFormula(
            storm::Environment const& env,
            storm::modelchecker::CheckTask<storm::logic::StateFormula, ValueType> const& checkTask) {
        storm::logic::StateFormula const& stateFormula = checkTask.getFormula();

        if (stateFormula.isProbabilityOperatorFormula()) {
            std::cout << "A new probability formula\n";
            this ->checkProbabilityOperatorFormula(
                    env,
                    checkTask.substituteFormula(stateFormula.asProbabilityOperatorFormula()));
        } else if (stateFormula.isBinaryStateFormula()) {
            std::cout << "A new binary formula\n";
        } else if (stateFormula.isUnaryBooleanStateFormula()) {
            std::cout << "A new unary boolean state formula" << std::endl;
        } else if (stateFormula.isBooleanLiteralFormula()) {
            std::cout << "A new boolean literal state formula" << std::endl;
            return this -> checkBooleanLiteralFormula(env, checkTask.substituteFormula(stateFormula.asBooleanLiteralFormula()));
        } else if (stateFormula.isAtomicExpressionFormula()) {
            std::cout << "A new atomic expression formula" << std::endl;
        } else if (stateFormula.isAtomicLabelFormula()) {
            std::cout << "A new atomic label formula" << std::endl;
            return this -> checkAtomicLabelFormula(env, stateFormula.asAtomicLabelFormula());
        } else {
            std::cout << "Something that is not handled yet" << std::endl;
        }
        throw std::runtime_error("checkStateFormula: Formula not implemented");
    }

    template <typename ValueType>
    void DTMCModelSolver<ValueType>::checkProbabilityOperatorFormula(
            const storm::Environment &env,
            const storm::modelchecker::CheckTask<storm::logic::ProbabilityOperatorFormula, ValueType> &checkTask) {
        storm::logic::ProbabilityOperatorFormula const& stateFormula = checkTask.getFormula();
        // call computeProbabilities here
        std::unique_ptr<storm::modelchecker::CheckResult> result = this ->computeProbabilities(env, checkTask.substituteFormula(stateFormula.getSubformula()));
    }

    template <typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::computeProbabilities(
        const storm::Environment &env,
        const storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> &checkTask) {
        storm::logic::Formula const& formula = checkTask.getFormula();

        std::cout << "Is state formula: " << (formula.isStateFormula() ? "yes" : "no")
            << ", Has qualitative Result " << (formula.hasQualitativeResult() ? "yes" : "no") << "\n";

        std::cout << "Is reachability formula? "
        << (formula.isReachabilityProbabilityFormula() ? "yes" : "no") << "\n";

        // basically before the until formula can be initiated there are some transformations
        // which need to be done. Then I think the model checking goes into a generic LTL checking
        // helper.
        
        if (formula.info(false).containsComplexPathFormula()) {
            // LTL Model Checking
            return this -> computeLTLProbabilities(
                env, checkTask.substituteFormula(formula.asPathFormula()));
        } else if (formula.isReachabilityProbabilityFormula()) {
            return this -> computeReachabilityProbabilities(
                env, checkTask.substituteFormula(formula.asReachabilityProbabilityFormula()));
        }
        throw std::runtime_error("checkProbabilityOperatorFormula: Not implemented");
    }

    template <typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::computeLTLProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> const& checkTask) {
        // todo
    }

    template <typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::computeReachabilityProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask){
        storm::logic::EventuallyFormula const& pathFormula = checkTask.getFormula();
        storm::logic::UntilFormula newFormula(storm::logic::Formula::getTrueFormula(), pathFormula.getSubformula().asSharedPointer());
        this -> computeUntilProbabilities(env, checkTask.substituteFormula(newFormula));
    }

    template <typename ValueType>
    void DTMCModelSolver<ValueType>::computeUntilProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::UntilFormula, ValueType> const& checkTask) {
        // First a recursive call on all of the left sub formulas probably until
        // they resolve to true or false or something like that. 
        storm::logic::UntilFormula const& pathFormula = checkTask.getFormula();
        storm::logic::Formula const& leftFormula = pathFormula.getLeftSubformula();
        std::cout << "Left formula: " << leftFormula << std::endl;
        std::cout << "Is the left formula a state formula?: " 
            << (leftFormula.isStateFormula() ? "yes" : "no") << "\n";
        std::unique_ptr<storm::modelchecker::CheckResult> leftResultPtr = this -> check(env, pathFormula.getLeftSubformula());
        std::unique_ptr<storm::modelchecker::CheckResult> rightResultPtr = this -> check(env, pathFormula.getRightSubformula());

        storm::modelchecker::ExplicitQualitativeCheckResult const& leftResult = leftResultPtr ->asExplicitQualitativeCheckResult();
        storm::modelchecker::ExplicitQualitativeCheckResult const& rightResult = rightResultPtr->asExplicitQualitativeCheckResult();

        storm::storage::BitVector const& phiStates = leftResult.getTruthValuesVector();
        storm::storage::BitVector const& psiStates = rightResult.getTruthValuesVector();

        storm::solver::GeneralLinearEquationSolverFactory<ValueType> linearEquationSolverFactory;
        bool convertToEquationSystem =
                linearEquationSolverFactory.getEquationProblemFormat(env) == storm::solver::LinearEquationSolverProblemFormat::EquationSystem;
        std::cout << "Convert to linear equations? " << (convertToEquationSystem ? "yes" : "no") << "\n";

        std::cout << "Is explicit model checker hint? " <<
            (checkTask.getHint().isExplicitModelCheckerHint() ? "yes" : "no\n");

        this->computeUntilProbabilitiesHelper(
            env,
            spModel.getInitialStates(),
            spModel,
            phiStates,
            psiStates,
            checkTask.getHint()
        );

        std::cout << "Reached the end without returning ptr, I should fail now bye 0/ \n";
    }

    template <typename ValueType>
    void DTMCModelSolver<ValueType>::computeUntilProbabilitiesHelper(
        storm::Environment const& env,
        storm::storage::BitVector const& relevantValues,
        mopmc::sparse::SparseModelBuilder<ValueType> &spModel,
        storm::storage::BitVector const& phiStates,
        storm::storage::BitVector const& psiStates,
        storm::modelchecker::ModelCheckerHint const& hint) {

        std::vector<ValueType> result(spModel.getNumberOfStates(), static_cast<ValueType>(0.0));
        storm::storage::BitVector maybeStates, statesWithProb1, statesWithProb0;

        if (hint.isExplicitModelCheckerHint() &&
            hint.template asExplicitModelCheckerHint<ValueType>().getComputeOnlyMaybeStates()){
        } else {
            // Get all the states that have probability 0 and 1 of satisfying the until formula
            std::cout << "computing the states with probability > 0\n";
            std::pair<storm::storage::BitVector, storm::storage::BitVector> statesWithProb01 =
                    mopmc::sparseutils::performProb01<ValueType>(spModel.getBackwardTransitions(),
                                                                 phiStates, psiStates);
            statesWithProb0 = std::move(statesWithProb01.first);
            statesWithProb1 = std::move(statesWithProb01.second);
            maybeStates = ~(statesWithProb0 | statesWithProb1);

            // set the exact values
            storm::utility::vector::setVectorValues<ValueType>(result, statesWithProb0, static_cast<ValueType>(0.));
            storm::utility::vector::setVectorValues<ValueType>(result, statesWithProb1, static_cast<ValueType>(1.));
        }

        bool maybeStatesNotRelevant = relevantValues.isDisjointFrom(maybeStates);

        if (maybeStatesNotRelevant) {
            std::cout << "States not relevant.\n";
        } else {
            if (!maybeStates.empty()) {
                // compute the probabilities

                // We are only going to be using Eigen meaning mopmc uses systems of equations
                // First get the correct submatrix
                // do (I - A)
                std::cout << "Constructing sub-matrix\n";
                std::cout << std::endl;
                
                std::pair<Eigen::SparseMatrix<ValueType>, std::unordered_map<uint_fast64_t, uint_fast64_t>>
                    returnPair = this -> spModel.getDTMCSubMatrix(maybeStates);
                
                std::cout << "Reduced Linear System:\n" << returnPair.first.toDense() << std::endl;
                std::cout << "Row size: " << returnPair.first.outerSize() << "\n";
                Eigen::VectorXd b = spModel.bVector(statesWithProb1,
                                spModel.getBackwardTransitions(),
                                returnPair.first.outerSize(),
                                returnPair.second);
                std::cout << "b: " << b.transpose() << std::endl;
                
                Eigen::VectorXd x = spModel.solverHelper(returnPair.first, b);
                
            }
        }
    }


    template <typename ValueType>
    void DTMCModelSolver<ValueType>::computeNextProbabilities(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask) {
        // todo
    }

    template <typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::checkBooleanLiteralFormula(
        storm::Environment const& env,
        storm::modelchecker::CheckTask<storm::logic::BooleanLiteralFormula, ValueType> const& checkTask
    ) {

        storm::logic::BooleanLiteralFormula const& stateFormula = checkTask.getFormula();
        if (stateFormula.isTrueFormula()) {
            return std::unique_ptr<storm::modelchecker::CheckResult>(
                new storm::modelchecker::ExplicitQualitativeCheckResult(
                        storm::storage::BitVector(spModel.getNumberOfStates(), true)));
        } else {
            return std::unique_ptr<storm::modelchecker::CheckResult>(
                new storm::modelchecker::ExplicitQualitativeCheckResult(
                        storm::storage::BitVector(spModel.getNumberOfStates())));
        }
    }


    template <typename ValueType>
    std::unique_ptr<storm::modelchecker::CheckResult> DTMCModelSolver<ValueType>::checkAtomicLabelFormula(
            const storm::Environment &env,
            const storm::modelchecker::CheckTask<storm::logic::AtomicLabelFormula, ValueType> &checkTask) {
        storm::logic::AtomicLabelFormula const& stateFormula = checkTask.getFormula();
        if (!spModel.hasLabel(stateFormula.getLabel())) {
            throw std::runtime_error("The property refers to an unknown label");
        }
        std::cout << "State Label Formula: " << stateFormula.getLabel() << std::endl;
        std::cout << "|States|: " << spModel.getStates(stateFormula.getLabel()).size() << std::endl;
        return std::unique_ptr<storm::modelchecker::CheckResult>(
            new storm::modelchecker::ExplicitQualitativeCheckResult(
                spModel.getStates(stateFormula.getLabel())
            )
        );
    }


    template class DTMCModelSolver<double>;
}
}

