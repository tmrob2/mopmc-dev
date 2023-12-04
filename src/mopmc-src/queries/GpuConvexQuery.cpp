//
// Created by guoxin on 3/11/23.
//


#include <iostream>
#include <Eigen/Dense>
#include "GpuConvexQuery.h"
#include "../solvers/ConvexQuery.h"
#include "../solvers/CudaValueIteration.cuh"
#include "../Data.h"
#include "../convex-functions/TotalReLU.h"
#include "../optimizers/FrankWolfe.h"
#include "../optimizers/ProjectedGradientDescent.h"
#include "../convex-functions/EuclideanDistance.h"
#include "../optimizers/PolytopeTypeEnum.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void GpuConvexQuery<T, I>::query() {

        mopmc::value_iteration::gpu::CudaValueIterationHandler<double> cudaVIHandler(
                this->data_.transitionMatrix,
                this->data_.rowGroupIndices,
                this->data_.row2RowGroupMapping,
                this->data_.flattenRewardVector,
                this->data_.defaultScheduler,
                this->data_.initialRow,
                this->data_.objectiveCount
        );
        cudaVIHandler.initialise();

        const uint64_t m = this->data_.objectiveCount; // m: number of objectives
        assert(this->data_.rowGroupIndices.size() == this->data_.colCount + 1);
        Vector<T> h = Eigen::Map<Vector<T>> (this->data_.thresholds.data(), this->data_.thresholds.size());

        std::vector<Vector<T>> Phi, W;
        Vector<T> vt = Vector<T>::Zero(m, 1), vb = Vector<T>::Zero(m, 1);
        Vector<T> *vPtr;
        Vector<T> r(m), w(m);
        //std::vector<T> r1;
        w.setConstant(static_cast<T>(1.0) / m);

        const double eps{1.e-6}, eps1{1.e-6};
        const uint_fast64_t maxIter{100};

        mopmc::optimization::convex_functions::EuclideanDistance<T> fn(h);
        mopmc::optimization::optimizers::FrankWolfe<T> frankWolfe(&fn);
        mopmc::optimization::optimizers::ProjectedGradientDescent<T> projectedGradientDescent(&fn);

        //Iteration
        uint_fast64_t iter = 0;
        T delta = 0;
        // at least iterate twice
        while (iter < maxIter && (Phi.size() < 3 || delta > eps)) {
            std::cout << "Main query loop: Iteration " << iter << "\n";
            if (!Phi.empty()) {
                vt = frankWolfe.argmin(Phi, *vPtr, Vertex, true);
                Vector <T> grad = fn.subgradient(vt);

                if (grad.template lpNorm<1>() < eps1) {
                    std::cout << "loop exit due to small gradient\n";
                    break;
                }
                w = static_cast<T>(-1.) * grad / grad.template lpNorm<1>();
            }

            // compute a new supporting hyperplane
            std::vector<T> w1(w.data(), w.data() + w.size());
            cudaVIHandler.valueIteration(w1);

            // get the first m elements of cudaVIHandler.results_
            std::vector<T> r1 = cudaVIHandler.getResults();
            r1.resize(m);
            r = VectorMap<T>(r1.data(), r1.size());
            Phi.push_back(r);
            W.push_back(w);

            if (Phi.size() == 1)
                vPtr = &r;
            else
                vPtr = &vt;

            if (W.size() == 1 || w.dot(r) < w.dot(vb)) {
                vb = projectedGradientDescent.argmin(*vPtr, Phi, W);
            }
            //delta = (vt - vb).template lpNorm<Eigen::Infinity>();
            delta = std::abs(fn.value(vt) - fn.value(vb));

            ++iter;
        }

        cudaVIHandler.exit();

        std::cout << "----------------------------------------------\n";
        std::cout << "CUDA CONVEX QUERY terminates after " << iter << " iteration(s) \n";
        std::cout << "*Threshold approximation* - upper bound: " << fn.value(vt)
                  << ", lower bound: " << fn.value(vb)
                  << ", gap: " << delta << "\n";
        std::cout << "upper bound point: [";
        for (int i = 0; i < m; ++i) {
            std::cout << vt(i) << " ";
        }
        std::cout << "]\n";
        std::cout << "lower bound point: [";
        for (int i = 0; i < m; ++i) {
            std::cout << vb(i) << " ";
        }
        std::cout << "]\n";
        std::cout << "----------------------------------------------\n";
    }

    template
    class GpuConvexQuery<double, int>;
}
