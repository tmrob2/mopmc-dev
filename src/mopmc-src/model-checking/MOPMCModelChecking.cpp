//
// Created by thomas on 6/09/23.
//
#include <storm/models/sparse/Mdp.h>
#include "MOPMCModelChecking.h"
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectiveRewardAnalysis.h>
#include <storm/transformer/EndComponentEliminator.h>
#include <storm/transformer/GoalStateMerger.h>
#include <storm/utility/vector.h>
#include <set>
#include <storm/solver/MinMaxLinearEquationSolver.h>
#include "../solvers/InducedEquationSolver.h"
#include "../solvers/IterativeSolver.h"
#include <storm/modelchecker/prctl/helper/DsMpiUpperRewardBoundsComputer.h>
#include <storm/modelchecker/prctl/helper/BaierUpperRewardBoundsComputer.h>
#include <storm/solver/LinearEquationSolver.h>
#include "../solvers/ConvexQuery.h"
#include <random>
#include "../solvers/SolverHelper.h"
#include "../solvers/CuVISolver.h"
#include <storm/modelchecker/helper/infinitehorizon/SparseNondeterministicInfiniteHorizonHelper.h>

namespace mopmc{

namespace multiobjective{

//! We reimplement check. This function call is a similar nomenclature to Storm however the way it is implemented is
//! different.
template <typename SparseModelType>
void MOPMCModelChecking<SparseModelType>::check(const storm::Environment &env,
                                                const std::vector<typename SparseModelType::ValueType> &weightVector) {
    // Prepare and invoke weighted infinite horizon (long run average) phase
    std::vector<typename SparseModelType::ValueType> weightedRewardVector(this->transitionMatrix.getRowCount(), storm::utility::zero<typename SparseModelType::ValueType>());
    if (!this->lraObjectives.empty()) {
        throw std::runtime_error("This framework does not deal with LRA.");
    }

    // Prepare and invoke weighted indefinite horizon (unbounded total reward) phase
    auto totalRewardObjectives = this->objectivesWithNoUpperTimeBound & ~this->lraObjectives;
    for (auto objIndex : totalRewardObjectives) {
        if (storm::solver::minimize(this->objectives[objIndex].formula->getOptimalityType())) {
            storm::utility::vector::addScaledVector(weightedRewardVector, this->actionRewards[objIndex], -weightVector[objIndex]);
        } else {
            storm::utility::vector::addScaledVector(weightedRewardVector, this->actionRewards[objIndex], weightVector[objIndex]);
        }
    }

    unboundedWeightedPhase(env, weightedRewardVector, weightVector);

    unboundedIndividualPhase(env, this->actionRewards, weightVector);
    //unboundedIndividualPhase(env, weightVector);

    std::cout << "initial state: " << this->initialState << "\n";
    std::cout << "result size: " << this->objectiveResults[0].size() << "\n";
    for (uint k = 0; k < this->objectives.size(); ++k) {
        std::cout << "Obj " << k << " " << this->objectiveResults[k][this->initialState] << "\n";
    }
}


//! This is Algorithm 1 in our paper.
template <typename SparseModelType>
void MOPMCModelChecking<SparseModelType>::multiObjectiveSolver(storm::Environment const& env) {
    // instantiate a random weight vector - to determine this we need to know the number of objectives


    uint64_t numObjs = this->objectives.size();
    // we also need to extract the constraints of the problem;
    typedef typename SparseModelType::ValueType T;

    // all objectives are equally weighted
    std::vector<typename SparseModelType::ValueType>
            w(numObjs, static_cast<typename SparseModelType::ValueType>(1.0) / static_cast<typename SparseModelType::ValueType>(numObjs));
    std::vector<std::vector<T>> W;
    std::vector<std::vector<T>> Phi;
    std::random_device rd; // random device for seed
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::vector<T> initialPoint(w.size());
    std::vector<T> constraints(w.size());
    std::vector<T> xStar(w.size(), std::numeric_limits<T>::max());
    std::set<std::vector<T>> wSet;

    // get thresholds from storm
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> c_(constraints.data(), constraints.size());
    for (uint64_t objIndex = 0; objIndex < w.size(); ++ objIndex) {
        constraints[objIndex] = this->objectives[objIndex].formula->template getThresholdAs<T>();
    }

    // set z* to the bounds of the problem.
    std::vector<T> zStar = constraints;
    std::vector<T> fxStar = mopmc::solver::convex::ReLU(xStar, constraints);
    std::vector<T> fzStar = mopmc::solver::convex::ReLU(zStar, constraints);
    T fDiff = mopmc::solver::convex::diff(fxStar, fzStar);

    uint64_t iteration = 0;
    do {
        if (!Phi.empty()) {
            std::cout << "Phi is not empty\n";
            // compute the FW and find a new weight vector
            xStar = mopmc::solver::convex::frankWolfe(mopmc::solver::convex::reluGradient<double>,
                                              initialPoint, 100, W, Phi, constraints);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> xStar_(xStar.data(), xStar.size());
            Eigen::Matrix<T, Eigen::Dynamic, 1> cx = c_ - xStar_;
            std::vector<T> grad = mopmc::solver::convex::reluGradient(cx);
            // If a w has already been seen before break;
            // make sure we call it w
            w = mopmc::solver::convex::computeNewW(grad);
            std::cout << "w*: ";
            for (int i = 0; i < w.size() ; ++i ){
                std::cout << w[i] << ",";
            }
            std::cout << "\n";

        }
        // if the w generated is already contained with W
        if (wSet.find(w) != wSet.end()) {
            std::cout << "W already in set => W ";
            for(auto val: w) {
                std::cout << val << ", ";
            }
            std::cout << "\n";
            break;
        } else {
            W.push_back(w);
            wSet.insert(w);
        }
        // compute a new supporting hyperplane
        std::cout << "w["<< iteration << "]: ";
        for (auto val: W[iteration]) {
            std::cout << val << ",";
        }
        std::cout << "\n";
        check(env, W.back());

        for (uint64_t objIndex = 0; objIndex < w.size(); ++objIndex) {
            std::cout << "Obj " << objIndex << " result " << this->objectiveResults[objIndex][this->initialState] << "\n";
            initialPoint[objIndex] = this->objectiveResults[objIndex][this->initialState];
        }

        Phi.push_back(initialPoint);
        // now we need to compute if the problem is still achievable
        T wr = std::inner_product(W[iteration].begin(), W[iteration].end(), Phi[iteration].begin(), static_cast<T>(0.));
        T wt = std::inner_product(W[iteration].begin(), W[iteration].end(), constraints.begin(), static_cast<T>(0.));

        std::cout << "wr: " << wr << ", " << "wt: " << wt << "\n";
        if (wr > wt) {
            // we now need to compute the point projection which minimises the convex function
            // if the problem is achievability just project the point a minimum distance away
            // to the nearest hyperplane.
            // randomly select an initial point from phi
            std::vector<T> projectedInitialPoint;
            if (Phi.size() > 1) {
                std::uniform_int_distribution<uint64_t> dist(0, Phi.size()); // Uniform distribution
                uint64_t randomIndex = dist(gen);
                projectedInitialPoint = Phi[randomIndex];
            } else {
                projectedInitialPoint = Phi[0];
            }

            T gamma = static_cast<T>(0.1);
            std::cout << "|Phi|: " << Phi.size() <<"\n";
            zStar = mopmc::solver::convex::projectedGradientDescent(
                    mopmc::solver::convex::reluGradient,
                    projectedInitialPoint,
                    gamma,
                    10,
                    Phi,
                    W,
                    Phi.size(),
                    constraints,
                    1.e-4);
        }
        ++iteration;

        std::cout << "x*: ";
        for (auto val: xStar) {
            std::cout << val <<", ";
        }
        std::cout << "\n";
        std::cout << "constraints: ";
        for (auto val: constraints) {
            std::cout << val <<", ";
        }
        std::cout << "\n";
        fxStar = mopmc::solver::convex::ReLU(xStar, constraints);
        fzStar = mopmc::solver::convex::ReLU(zStar, constraints);
        std::cout << "f(x*): ";
        for (auto val: fxStar) {
            std::cout << val <<", ";
        }
        std::cout << "f(z*): ";
        for (auto val: fzStar) {
            std::cout << val <<", ";
        }
        fDiff = mopmc::solver::convex::diff(fxStar, fzStar);
        std::cout << " L1Norm: " << fDiff << "\n";
    } while (fDiff > static_cast<T>(0.));
}

template<class SparseModelType>
void MOPMCModelChecking<SparseModelType>::unboundedWeightedPhase(storm::Environment const& env,
                                                                 std::vector<typename SparseModelType::ValueType> const& weightedRewardVector,
                                                                 std::vector<typename SparseModelType::ValueType> const& weightVector) {
    // Catch the case where all values on the RHS of the MinMax equation system are zero.
    if (this->objectivesWithNoUpperTimeBound.empty() ||
        ((this->lraObjectives.empty() || !storm::utility::vector::hasNonZeroEntry(this->lraMecDecomposition->auxMecValues)) &&
         !storm::utility::vector::hasNonZeroEntry(weightedRewardVector))) {
        this->weightedResult.assign(this->transitionMatrix.getRowGroupCount(), storm::utility::zero<typename SparseModelType::ValueType>());
        storm::storage::BitVector statesInLraMec(this->transitionMatrix.getRowGroupCount(), false);
        if (this->lraMecDecomposition) {
            for (auto const& mec : this->lraMecDecomposition->mecs) {
                for (auto const& sc : mec) {
                    statesInLraMec.set(sc.first, true);
                }
            }
        }
        // Get an arbitrary scheduler that yields finite reward for all objectives
        this->computeSchedulerFinitelyOften(this->transitionMatrix, this->transitionMatrix.transpose(true), ~this->actionsWithoutRewardInUnboundedPhase, statesInLraMec,
                                      this->optimalChoices);
        return;
    }

    //! This is a storm function. We use it to compute and collapse the End Component sub model based on the weighted
    //! rewards model (w.r).
    this->updateEcQuotient(weightedRewardVector);

    // Set up the choice values
    storm::utility::vector::selectVectorValues(this->ecQuotient->auxChoiceValues, this->ecQuotient->ecqToOriginalChoiceMapping, weightedRewardVector);
    std::map<uint64_t, uint64_t> ecqStateToOptimalMecMap;
    if (!this->lraObjectives.empty()) {
        throw std::runtime_error("This framework does not deal with LRA");
    }

    std::vector<uint64_t > scheduler = this->computeValidInitialScheduler(this->ecQuotient->matrix, this->ecQuotient->rowsWithSumLessOne);
    Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1> b(this->ecQuotient->matrix.getRowGroupCount());

    Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> x(this->ecQuotient->auxStateValues.data(), this->ecQuotient->auxStateValues.size());
    std::cout << "|b|: " << b.size() << "\n";
    reduceMatrixToDTMC(b, scheduler);
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> I = makeEigenIdentityMatrix();
    // compute the state value vector for the initial scheduler
    mopmc::solver::linsystem::solverHelper(b, x, eigenTransitionMatrix, I);

    // convert the transition matrix to a sparse eigen matrix for VI
    toEigenSparseMatrix();
    //std::cout << eigenTransitionMatrix << std::endl;

    // This computation can be done on the GPU.
    mopmc::solver::iter::valueIteration(this->eigenTransitionMatrix, x, this->ecQuotient->auxChoiceValues,
                                        scheduler, this->ecQuotient->matrix.getRowGroupIndices());
    std::vector<int> rowGroupIndices(this->ecQuotient->matrix.getRowGroupIndices().begin(), this->ecQuotient->matrix.getRowGroupIndices().end());
    std::vector<int> scheduler2(scheduler.begin(), scheduler.end());
    mopmc::solver::cuda::valueIteration(eigenTransitionMatrix, this->ecQuotient->auxStateValues,this->ecQuotient->auxChoiceValues,
                                        scheduler2, rowGroupIndices);
    std::transform(scheduler2.begin(), scheduler2.end(),
                   scheduler.begin(), [](int x){ return static_cast<uint_fast64_t>(x);});

    this->weightedResult = std::vector<typename SparseModelType::ValueType>(this->transitionMatrix.getRowGroupCount());
    this->transformEcqSolutionToOriginalModel(this->ecQuotient->auxStateValues, scheduler, ecqStateToOptimalMecMap,
                                        this->weightedResult, this->optimalChoices);
}

template <typename SparseModelType>
void MOPMCModelChecking<SparseModelType>::unboundedIndividualPhase(const storm::Environment &env,
                                                                   std::vector<std::vector<typename SparseModelType::ValueType>> &rewardModels,
                                                                   const std::vector<typename SparseModelType::ValueType> &weightVector) {
    // First step is to create a deterministic sparse matrix from the scheduler

    // We then also want to set up the DTMC equation system so we need to do some further graph analysis

    storm::storage::SparseMatrix<typename SparseModelType::ValueType> deterministicMatrix =
            this->transitionMatrix.selectRowsFromRowGroups(this->optimalChoices, false);

    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> eigenDeterministicMatrix =
            mopmc::solver::helper::eigenInducedTransitionMatrix<SparseModelType>(
                    this->transitionMatrix, this->optimalChoices,
                    this->transitionMatrix.getRowGroupIndices());
    /*Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> testMatrix(deterministicMatrix.getRowCount(), deterministicMatrix.getColumnCount());
    for(uint_fast64_t i = 0; i < deterministicMatrix.getRowCount(); ++i) {
        for(const auto& entry: deterministicMatrix.getRow(i)) {
            testMatrix.insert(i, entry.getColumn()) = entry.getValue();
        }
    }

    testMatrix.makeCompressed();*/

    //std::cout << "Matrix equal " << (testMatrix.toDense() == eigenDeterministicMatrix.toDense() ? "yes" : "no") << "\n";

    std::vector<typename SparseModelType::ValueType> deterministicStateRewards(deterministicMatrix.getRowCount());  // allocate here

    // We compute an estimate for the results of the individual objectives which is obtained from the weighted result and the result of the objectives
    // computed so far. Note that weightedResult = Sum_{i=1}^{n} w_i * objectiveResult_i.
    std::vector<typename SparseModelType::ValueType> weightedSumOfUncheckedObjectives = weightedResult;


    Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, Eigen::Dynamic> R(deterministicStateRewards.size(), weightVector.size());

    for (uint_fast64_t const& objIndex : storm::utility::vector::getSortedIndices(weightVector)) {
        auto const& obj = this->objectives[objIndex];
        // Make sure that the objectiveResult is initialized correctly
        this->objectiveResults[objIndex].resize(this->transitionMatrix.getRowGroupCount(), storm::utility::zero<typename SparseModelType::ValueType>());
        std::cout << "Got here!\n";
        storm::utility::vector::selectVectorValues(deterministicStateRewards, this->optimalChoices, this->transitionMatrix.getRowGroupIndices(),
                                                   this->actionRewards[objIndex]);
        //storm::storage::BitVector statesWithRewards = ~storm::utility::vector::filterZero(deterministicStateRewards);
        Eigen::Map<Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1>> col(deterministicStateRewards.data(), deterministicStateRewards.size());
        R.col(objIndex) = col;
    }
    Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, Eigen::Dynamic> X =
            Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, Eigen::Dynamic>::Zero(deterministicStateRewards.size(), weightVector.size());
    mopmc::solver::iter::objValueIteration(eigenDeterministicMatrix, X, R);
    for (uint_fast64_t const& objIndex: storm::utility::vector::getSortedIndices(weightVector)) {
        storm::utility::vector::setVectorValues<typename SparseModelType::ValueType>(
            this->objectiveResults[objIndex],
            storm::storage::BitVector(X.rows(), true),
            std::vector<typename SparseModelType::ValueType>(
                    X.col(objIndex).data(), X.col(objIndex).data() + X.col(objIndex).size()));
    }
}

//! This is a storm function that we might need for runtime clock testing
/*template<typename SparseModelType>
void MOPMCModelChecking<SparseModelType>::unboundedIndividualPhase(const storm::Environment &env,
                                                                   const std::vector<typename SparseModelType::ValueType> &weightVector) {
    std::cout << "objectives with no upper time bound: " << this->objectivesWithNoUpperTimeBound.getNumberOfSetBits() << "\n";
    std::cout << "weight 1: " << storm::utility::isOne(weightVector[*this->objectivesWithNoUpperTimeBound.begin()]) << "\n";
    // we will never have objectives with upper time bounds in this framework

    if (this->objectivesWithNoUpperTimeBound.getNumberOfSetBits() == 1 && storm::utility::isOne(weightVector[*this->objectivesWithNoUpperTimeBound.begin()])) {
        uint_fast64_t objIndex = *this->objectivesWithNoUpperTimeBound.begin();
        objectiveResults[objIndex] = weightedResult;
        if (storm::solver::minimize(this->objectives[objIndex].formula->getOptimalityType())) {
            storm::utility::vector::scaleVectorInPlace(objectiveResults[objIndex], -storm::utility::one<typename SparseModelType::ValueType>());
        }
        for (uint_fast64_t objIndex2 = 0; objIndex2 < this->objectives.size(); ++objIndex2) {
            if (objIndex != objIndex2) {
                objectiveResults[objIndex2] = std::vector<typename SparseModelType::ValueType>(this->transitionMatrix.getRowGroupCount(), storm::utility::zero<typename SparseModelType::ValueType>());
            }
        }
    } else {
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> deterministicMatrix =
                this->transitionMatrix.selectRowsFromRowGroups(this->optimalChoices, false);
        storm::storage::SparseMatrix<typename SparseModelType::ValueType> deterministicBackwardTransitions =
                deterministicMatrix.transpose();
        std::vector<typename SparseModelType::ValueType> deterministicStateRewards(deterministicMatrix.getRowCount());  // allocate here
        storm::solver::GeneralLinearEquationSolverFactory<typename SparseModelType::ValueType> linearEquationSolverFactory;

        auto infiniteHorizonHelper = this->createDetInfiniteHorizonHelper(deterministicMatrix);
        infiniteHorizonHelper.provideBackwardTransitions(deterministicBackwardTransitions);

        // We compute an estimate for the results of the individual objectives which is obtained from the weighted result and the result of the objectives
        // computed so far. Note that weightedResult = Sum_{i=1}^{n} w_i * objectiveResult_i.
        std::vector<typename SparseModelType::ValueType> weightedSumOfUncheckedObjectives = weightedResult;
        typename SparseModelType::ValueType sumOfWeightsOfUncheckedObjectives =
                storm::utility::vector::sum_if(weightVector, objectivesWithNoUpperTimeBound);

        for (uint_fast64_t const& objIndex : storm::utility::vector::getSortedIndices(weightVector)) {
            auto const& obj = this->objectives[objIndex];
            if (objectivesWithNoUpperTimeBound.get(objIndex)) {
                offsetsToUnderApproximation[objIndex] = storm::utility::zero<typename SparseModelType::ValueType>();
                offsetsToOverApproximation[objIndex] = storm::utility::zero<typename SparseModelType::ValueType>();
                if (lraObjectives.get(objIndex)) {
                    auto actionValueGetter = [&](uint64_t const& a) {
                        return actionRewards[objIndex][transitionMatrix.getRowGroupIndices()[a] + this->optimalChoices[a]];
                    };
                    typename storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<typename SparseModelType::ValueType>::ValueGetter stateValueGetter;
                    if (stateRewards.empty() || stateRewards[objIndex].empty()) {
                        stateValueGetter = [](uint64_t const&) { return storm::utility::zero<typename SparseModelType::ValueType>(); };
                    } else {
                        stateValueGetter = [&](uint64_t const& s) { return stateRewards[objIndex][s]; };
                    }
                    objectiveResults[objIndex] = infiniteHorizonHelper.computeLongRunAverageValues(env, stateValueGetter, actionValueGetter);
                } else {  // i.e. a total reward objective
                    storm::utility::vector::selectVectorValues(deterministicStateRewards, this->optimalChoices, transitionMatrix.getRowGroupIndices(),
                                                               actionRewards[objIndex]);
                    storm::storage::BitVector statesWithRewards = ~storm::utility::vector::filterZero(deterministicStateRewards);
                    // As maybestates we pick the states from which a state with reward is reachable
                    storm::storage::BitVector maybeStates = storm::utility::graph::performProbGreater0(
                            deterministicBackwardTransitions, storm::storage::BitVector(deterministicMatrix.getRowCount(), true), statesWithRewards);

                    // Compute the estimate for this objective
                    if (!storm::utility::isZero(weightVector[objIndex])) {
                        objectiveResults[objIndex] = weightedSumOfUncheckedObjectives;
                        typename SparseModelType::ValueType scalingFactor = storm::utility::one<typename SparseModelType::ValueType>() / sumOfWeightsOfUncheckedObjectives;
                        if (storm::solver::minimize(obj.formula->getOptimalityType())) {
                            scalingFactor *= -storm::utility::one<typename SparseModelType::ValueType>();
                        }
                        storm::utility::vector::scaleVectorInPlace(objectiveResults[objIndex], scalingFactor);
                        storm::utility::vector::clip(objectiveResults[objIndex], obj.lowerResultBound, obj.upperResultBound);
                    }
                    // Make sure that the objectiveResult is initialized correctly
                    objectiveResults[objIndex].resize(transitionMatrix.getRowGroupCount(), storm::utility::zero<typename SparseModelType::ValueType>());

                    if (!maybeStates.empty()) {
                        bool needEquationSystem =
                                linearEquationSolverFactory.getEquationProblemFormat(env) == storm::solver::LinearEquationSolverProblemFormat::EquationSystem;
                        storm::storage::SparseMatrix<typename SparseModelType::ValueType> submatrix =
                                deterministicMatrix.getSubmatrix(true, maybeStates, maybeStates, needEquationSystem);
                        if (needEquationSystem) {
                            // Converting the matrix from the fixpoint notation to the form needed for the equation
                            // system. That is, we go from x = A*x + b to (I-A)x = b.
                            submatrix.convertToEquationSystem();
                        }

                        // Prepare solution vector and rhs of the equation system.
                        std::vector<typename SparseModelType::ValueType> x = storm::utility::vector::filterVector(objectiveResults[objIndex], maybeStates);
                        std::vector<typename SparseModelType::ValueType> b = storm::utility::vector::filterVector(deterministicStateRewards, maybeStates);

                        // Now solve the resulting equation system.
                        std::unique_ptr<storm::solver::LinearEquationSolver<typename SparseModelType::ValueType>> solver = linearEquationSolverFactory.create(env, submatrix);
                        auto req = solver->getRequirements(env);
                        solver->clearBounds();
                        storm::storage::BitVector submatrixRowsWithSumLessOne = deterministicMatrix.getRowFilter(maybeStates, maybeStates) % maybeStates;
                        submatrixRowsWithSumLessOne.complement();
                        this->setBoundsToSolver(*solver, req.lowerBounds(), req.upperBounds(), objIndex, submatrix, submatrixRowsWithSumLessOne, b);
                        if (solver->hasLowerBound()) {
                            req.clearLowerBounds();
                        }
                        if (solver->hasUpperBound()) {
                            req.clearUpperBounds();
                        }
                        //STORM_LOG_THROW(!req.hasEnabledCriticalRequirement(), storm::exceptions::UncheckedRequirementException,
                        //                "Solver requirements " + req.getEnabledRequirementsAsString() + " not checked.");
                        solver->solveEquations(env, x, b);
                        // Set the result for this objective accordingly
                        storm::utility::vector::setVectorValues<typename SparseModelType::ValueType>(objectiveResults[objIndex], maybeStates, x);
                    }
                    storm::utility::vector::setVectorValues<typename SparseModelType::ValueType>(objectiveResults[objIndex], ~maybeStates, storm::utility::zero<typename SparseModelType::ValueType>());
                }
                // Update the estimate for the next objectives.
                if (!storm::utility::isZero(weightVector[objIndex])) {
                    storm::utility::vector::addScaledVector(weightedSumOfUncheckedObjectives, objectiveResults[objIndex], -weightVector[objIndex]);
                    sumOfWeightsOfUncheckedObjectives -= weightVector[objIndex];
                }
            } else {
                objectiveResults[objIndex] = std::vector<typename SparseModelType::ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<typename SparseModelType::ValueType>());
            }
        }
    }

    for (uint64_t k = 0; k < objectives.size(); ++k ) {
        for (uint_fast64_t i = 0; i < objectiveResults[k].size(); ++i){
            std::cout << objectiveResults[k][i] << " ";
        }
        std::cout << "\n\n";
    }
}
*/

template<typename SparseModelType>
Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> MOPMCModelChecking<SparseModelType>::makeEigenIdentityMatrix() {
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> I (this->ecQuotient->matrix.getRowGroupCount(),
                                                                                 this->ecQuotient->matrix.getRowGroupCount());
    for (uint_fast64_t i = 0; i < this->ecQuotient->matrix.getRowGroupCount(); ++i) {
        I.insert(i, i) = static_cast<typename SparseModelType::ValueType>(1.0);
    }
    I.finalize();
    return I;
}

template<typename SparseModelType>
void MOPMCModelChecking<SparseModelType>::toEigenSparseMatrix() {
    std::vector<Eigen::Triplet<typename SparseModelType::ValueType>> triplets;
    triplets.reserve(this->ecQuotient->matrix.getNonzeroEntryCount());

    for(uint_fast64_t row = 0; row < this->ecQuotient->matrix.getRowCount(); ++row) {
        for(auto element : this->ecQuotient->matrix.getRow(row)) {
            //std::cout << "row: " << row << " col: " << element.getColumn() << " val: " << element.getValue() << "\n";
            triplets.emplace_back(row, element.getColumn(), element.getValue());
        }
    }

    Eigen::SparseMatrix<typename SparseModelType::ValueType,  Eigen::RowMajor> result =
            Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor>(
                    this->ecQuotient->matrix.getRowCount(), this->ecQuotient->matrix.getColumnCount()
            );
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    this->eigenTransitionMatrix = std::move(result);
}

template <typename SparseModelType>
void MOPMCModelChecking<SparseModelType>::reduceMatrixToDTMC(
        Eigen::Matrix<typename SparseModelType::ValueType, Eigen::Dynamic, 1> &b,
        std::vector<uint64_t> const& scheduler) {
    std::cout << "Storm matrix: " << this->ecQuotient->matrix.getRowGroupCount() << "\n";
    std::cout << "EC num choices: " << eigenTransitionMatrix.rows() << "\n";
    std::cout << "Scheduler size: " << scheduler.size() << "\n";
    auto rowGroupIndexIt = this->ecQuotient->matrix.getRowGroupIndices().begin();
    uint_fast64_t n = this->ecQuotient->matrix.getRowGroupCount();
    std::cout << "Matrix dimensions: (" << n << ", " << n <<")\n";
    std::cout << "Ecq choice values size: " << this->ecQuotient->auxChoiceValues.size() << "\n";
    auto policyChoiceIt = scheduler.begin();
    SpMat subMatrix(n, n);
    for (uint_fast64_t state = 0; state < n; ++state ) {
        uint_fast64_t row = (*rowGroupIndexIt) + (*policyChoiceIt);

        b[state] = this->ecQuotient->auxChoiceValues[row];

        ++rowGroupIndexIt;
        ++policyChoiceIt;
        for(auto const& entry : this->ecQuotient->matrix.getRow(row)) {
            //std::cout << "state: " << state << " action: " << *policyChoiceIt <<
            //          " row: " << state << " col: " << entry.getColumn() << " value: " << entry.getValue() << "\n";
            subMatrix.insert(state, entry.getColumn()) = entry.getValue();
        }
    }
    subMatrix.makeCompressed();
    this->eigenTransitionMatrix = std::move(subMatrix);
}

template class MOPMCModelChecking<storm::models::sparse::Mdp<double>>;

}
}
