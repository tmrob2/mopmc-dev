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
#include "ModelBuilding.h"
#include "Transformation.h"
#include "Data.h"
#include "queries/GpuConvexQuery.h"
#include "queries/GpuConvexQueryAlt.h"
#include "queries/AchievabilityQuery.h"
#include "convex-functions/TotalReLU.h"
#include "convex-functions/SignedKLEuclidean.h"
#include "queries/TestingQuery.h"
#include "lp_lib.h"
#include "StormModelCheckingWrapper.h"
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
        //mopmc::queries::GpuConvexQuery<ValueType, int> q(data);
        mopmc::queries::GpuConvexQueryAlt<ValueType, int> q(data);
        //mopmc::queries::TestingQuery<ValueType, int> q(data);
        //mopmc::queries::AchievabilityQuery<ValueType, int> q(data);
        q.query();
        clock_t time3 = clock();

        std::cout<<"       TIME STATISTICS        \n";
        printf("Model building stage 1: %.3f seconds.\n", double(time05 - time0)/CLOCKS_PER_SEC);
        printf("Model building stage 2: %.3f seconds.\n", double(time1 - time05)/CLOCKS_PER_SEC);
        printf("Input data transformation: %.3f seconds.\n", double(time2 - time1)/CLOCKS_PER_SEC);
        printf("Model checking: %.3f seconds.\n", double(time3 - time2)/CLOCKS_PER_SEC);
        /*
        std::vector<double> c = {2.0, 0.5};
        std::vector<bool> b = {false, true};
        std::vector<double> x = {1.8, 0.42};

        mopmc::optimization::convex_functions::SignedKLEuclidean<double> kle(c, b);
        std::cout << "## std::vector results: \n";
        std::cout << "kle value1 at x = {1.8, 0.42}: " << kle.value1(x) << "\n";
        std::cout << "kle gradient at x = {1.8, 0.42}: [ " << kle.subgradient1(x)[0] << ", " << kle.subgradient1(x)[1] << "]\n";
        std::cout << "kle value1 at c = {2.0, 0.5}: " << kle.value1(c) << "\n";
        std::cout << "kle gradient c = {2.0, 0.5}: [ " << kle.subgradient1(c)[0] << ", " << kle.subgradient1(c)[1] << "]\n";

        using Vector =  Eigen::Matrix<double, Eigen::Dynamic, 1>;
        using VectorMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>;

        Vector c_ = VectorMap(c.data(), c.size());
        Vector x_ = VectorMap(x.data(), x.size());
        std::cout << "## Eigen vector results: \n";
        mopmc::optimization::convex_functions::SignedKLEuclidean<double> kle1(c_, b);
        std::cout << "kle value1 at x = {1.8, 0.42}: " << kle1.value(x_) << "\n";
        std::cout << "kle gradient at x = {1.8, 0.42}: [ " << kle1.subgradient(x_)(0) << ", " << kle1.subgradient(x_)(1) << "]\n";
        std::cout << "kle value1 at c = {2.0, 0.5}: " << kle1.value(c_) << "\n";
        std::cout << "kle gradient c = {2.0, 0.5}: [ " << kle1.subgradient(c_)(0) << ", " << kle1.subgradient(c_)(1) << "]\n";
         */

        return true;

    }
}