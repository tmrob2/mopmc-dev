//
// Created by guoxin on 2/11/23.
//


#include <storm-parsers/api/storm-parsers.h>
#include <string>
#include <iostream>
#include <storm/environment/modelchecker/MultiObjectiveModelCheckerEnvironment.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/modelchecker/multiobjective/pcaa/StandardMdpPcaaWeightVectorChecker.h>
#include <Eigen/Sparse>
#include "Runner.h"
//#include "ExplicitModelBuilder.h"
#include "mopmc-src/storm-wrappers/StormModelBuildingWrapper.h"
#include "Transformation.h"
#include "mopmc-src/hybrid-computing/Problem.h"
#include "queries/ConvexQuery.h"
#include "queries/AchievabilityQuery.h"
#include "convex-functions/TotalReLU.h"
#include "convex-functions/SignedKLEuclidean.h"
#include "queries/TestingQuery.h"
#include "convex-functions/EuclideanDistance.h"
#include "optimizers/FrankWolfe.h"
#include "optimizers/ProjectedGradientDescent.h"
#include "mopmc-src/storm-wrappers/StormModelCheckingWrapper.h"
#include <Eigen/Dense>
#include <cstdio>
#include <ctime>

namespace mopmc {

    typedef storm::models::sparse::Mdp<double> ModelType;
    typedef storm::models::sparse::Mdp<double>::ValueType ValueType;
    typedef storm::storage::sparse::state_type IndexType;
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType> PreprocessedType;
    typedef storm::modelchecker::multiobjective::preprocessing::SparseMultiObjectivePreprocessor<ModelType>::ReturnType PrepReturnType;
    typedef Eigen::SparseMatrix<ValueType, Eigen::RowMajor> EigenSpMatrix;

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    bool run(std::string const &path_to_model, std::string const &property_string) {

        assert (typeid(ValueType)==typeid(double));
        assert (typeid(IndexType)==typeid(uint64_t));

        storm::Environment env;
        clock_t time0 = clock();
        auto preprocessedResult = mopmc::ModelBuilder<ModelType>::preprocess(path_to_model, property_string, env);
        clock_t time05 = clock();
        //mopmc::wrapper::StormModelCheckingWrapper<ModelType> stormModelCheckingWrapper(preprocessedResult);
        //stormModelCheckingWrapper.performMultiObjectiveModelChecking(env);
        auto preparedModel = mopmc::ModelBuilder<ModelType>::build(preprocessedResult);
        clock_t time1 = clock();
        auto data = mopmc::Transformation<ModelType, ValueType, IndexType>::transform_i32_v2(preprocessedResult, preparedModel);
        clock_t time2 = clock();

        //threshold
        auto h = Eigen::Map<Vector<ValueType >> (data.thresholds.data(), data.thresholds.size());
        //convex functon
        mopmc::optimization::convex_functions::EuclideanDistance<ValueType> fn(h);
        //optimizers
        mopmc::optimization::optimizers::FrankWolfe<ValueType> frankWolfe(mopmc::optimization::optimizers::FWOptMethod::LINOPT, &fn);
        mopmc::optimization::optimizers::ProjectedGradientDescent<ValueType> projectedGD(
                mopmc::optimization::optimizers::ProjectionType::NearestHyperplane, &fn);
        mopmc::optimization::optimizers::ProjectedGradientDescent<ValueType> projectedGD1(
                mopmc::optimization::optimizers::ProjectionType::UnitSimplex, &fn);
        //value-iteration solver
        mopmc::value_iteration::gpu::CudaValueIterationHandler<double> cudaVIHandler(
                data.transitionMatrix,
                data.rowGroupIndices,
                data.row2RowGroupMapping,
                data.flattenRewardVector,
                data.defaultScheduler,
                data.initialRow,
                data.objectiveCount
        );

        mopmc::queries::ConvexQuery<ValueType, int> q(data, &fn, &frankWolfe, &projectedGD, &cudaVIHandler);
        //mopmc::queries::ConvexQuery<ValueType, int> q(data, &fn, &projectedGD1, &projectedGD, &cudaVIHandler);
        //mopmc::queries::TestingQuery<ValueType, int> q(data, &fn, &projectedGD1, &projectedGradientDescent);
        //mopmc::queries::AchievabilityQuery<ValueType, int> q(data);
        q.query();
        //q.hybridQuery(hybrid::ThreadSpecialisation::GPU);
        clock_t time3 = clock();

        std::cout<<"       TIME STATISTICS        \n";
        printf("Model building stage 1: %.3f seconds.\n", double(time05 - time0)/CLOCKS_PER_SEC);
        printf("Model building stage 2: %.3f seconds.\n", double(time1 - time05)/CLOCKS_PER_SEC);
        printf("Input data transformation: %.3f seconds.\n", double(time2 - time1)/CLOCKS_PER_SEC);
        printf("Model checking: %.3f seconds.\n", double(time3 - time2)/CLOCKS_PER_SEC);

        return true;

    }
}