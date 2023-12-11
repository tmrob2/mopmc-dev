//
// Created by guoxin on 27/11/23.
//

#include "TestingQuery.h"
#include "../convex-functions/TotalReLU.h"
#include "../optimizers/FrankWolfe.h"
#include "../optimizers/ProjectedGradientDescent.h"
#include "../convex-functions/EuclideanDistance.h"
#include <iostream>
#include <Eigen/Dense>

namespace mopmc::queries {

    template<typename T, typename I>
    void TestingQuery<T, I>::query() {

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
        Vector<T> h = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());


        std::vector<Vector<T>> Phi, W;
        Vector<T> vt = Vector<T>::Zero(m, 1), vb = Vector<T>::Zero(m, 1);
        Vector<T> vPrv;
        Vector<T> r(m), w(m), q;

        const double eps{1.e-6}, eps1{1.e-8}, eps2{1.e-6};
        const uint_fast64_t maxIter{100};
        T tol, tol1, tol2;
        uint_fast64_t iter = 0;


        while (iter < maxIter) {
            std::cout << "Main loop: Iteration " << iter << "\n";
            if (!Phi.empty()) {

                this->primaryOptimizer->minimize(vt, Phi);
                Vector<T> grad = this->fn->subgradient(vt);

                tol1 = grad.template lpNorm<1>();
                if (tol1 < eps1) {
                    std::cout << "loop exit due to small gradient (tolerance: " << tol1 <<")\n";
                    ++iter;
                    break;
                }
                w = static_cast<T>(-1.) * grad / grad.template lpNorm<1>();
            } else {
                w.setConstant(static_cast<T>(1.0) / m);
            }

            // compute a new supporting hyperplane
            std::vector<T> w1(w.data(), w.data() + w.size());
            cudaVIHandler.valueIteration(w1);

            std::vector<T> r1 = cudaVIHandler.getResults();
            // only need the first m elements
            r1.resize(m);
            r = VectorMap<T>(r1.data(), r1.size());

            if (Phi.size() >= 2) {
                //err2 = (vt - vPrv).template lpNorm<1>();
                tol2 = (vt - vPrv).template lpNorm<Eigen::Infinity>();
                if (tol2 < eps2) {
                    std::cout << "loop exit due to small improvement on (estimated) nearest point (tolerance: " << tol2 <<")\n";
                    ++iter;
                    break;
                }
            }

            Phi.push_back(r);
            W.push_back(w);

            if (Phi.size() == 1) {
                //vPtr = &r;
                vPrv = r;
            } else {
                //vPtr = &vt;
                vPrv = vt;
            }

            if (W.size() == 1 || w.dot(r) < w.dot(vb)) {
                vb = vt;
                this->secondaryOptimizer->minimize(vb, Phi, W);
            }
            //tol = (vt - vb).template lpNorm<Eigen::Infinity>();
            tol = std::abs(this->fn->value(vt) - this->fn->value(vb));
            if (tol < eps) {
                std::cout << "loop exit due to small distance on threshold (tolerance: " << tol <<")\n";
                ++iter;
                break;
            }

            ++iter;
        }

        cudaVIHandler.exit();

        {
            Vector<T> vOut = (vb + vt) * static_cast<T>(0.5);
            std::cout << "----------------------------------------------\n"
                      << "CUDA CONVEX QUERY terminates after " << iter << " iteration(s)\n"
                      //<< "Tolerance: " << tol << ", Tolerance1: " << tol1 << ", Tolerance2: " << tol2 << "\n"
                      << "Estimated nearest point to threshold : [";
            for (int i = 0; i < m; ++i) {
                std::cout << vt(i) << " ";
            }
            std::cout << "]\n"
                      << "Approximate distance: " << this->fn->value(vt)
                      //<< "Approximate distance between " << fn.value(vb) << " and " << fn.value(vt)
                      << "\n----------------------------------------------\n";
        }
    }

    template
    class TestingQuery<double, int>;
}