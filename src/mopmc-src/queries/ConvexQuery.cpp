//
// Created by guoxin on 3/11/23.
//


#include <iostream>
#include <Eigen/Dense>
#include "../solvers/WarmUp.h"
#include "ConvexQuery.h"
#include "../solvers/CudaValueIteration.cuh"
#include "../Data.h"
#include "../convex-functions/TotalReLU.h"
#include "../optimizers/FrankWolfe.h"
#include "../optimizers/ProjectedGradientDescent.h"
#include "../convex-functions/EuclideanDistance.h"
#include "../optimizers/PolytopeTypeEnum.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void ConvexQuery<T, I>::query() {

        //where is the best place to warm up gpu?
        mopmc::kernels::launchWarmupKernel();
        this->VIhandler->initialize();
        const uint64_t m = this->data_.objectiveCount; // m: number of objectives
        assert(this->data_.rowGroupIndices.size() == this->data_.colCount + 1);
        Vector<T> h = Eigen::Map<Vector<T>> (this->data_.thresholds.data(), this->data_.thresholds.size());

        std::vector<Vector<T>> Phi, W;
        Vector<T> vt = Vector<T>::Zero(m, 1), vb = Vector<T>::Zero(m, 1);
        Vector<T> vPrv;
        Vector<T> r(m), w(m);
        w.setConstant(static_cast<T>(1.0) / m);

        const double eps{1.e-6}, eps1{1.e-8}, eps2{1.e-6};
        const uint_fast64_t maxIter{100};
        T tol, tol1, tol2;
        uint_fast64_t iter = 0;

        while (iter < maxIter) {
            std::cout << "Main loop: Iteration " << iter << "\n";
            //std::cout << "Tolerance: " << tol << ", Tolerance1: " << tol1 << ", Tolerance2: " << tol2 << "\n";
            if (!Phi.empty()) {
                vt = vPrv;
                this->primaryOptimizer->minimize(vt, Phi);

                if (Phi.size() >= 2) {
                    //tol2 = (vPrv - vt).template lpNorm<1>();
                    tol2 = (vPrv - vt).template lpNorm<Eigen::Infinity>();
                    if (tol2 < eps2) {
                        std::cout << "loop exit due to small improvement on (estimated) nearest point (tolerance: " << tol2 << ")\n";
                        ++iter;
                        break;
                    }
                }

                Vector<T> grad = this->fn->subgradient(vt);
                tol1 = grad.template lpNorm<1>();
                if (tol1 < eps1) {
                    std::cout << "loop exit due to small gradient (tolerance: " << tol1 << ")\n";
                    ++iter;
                    break;
                }
                w = static_cast<T>(-1.) * grad / grad.template lpNorm<1>();
            }

            // compute a new supporting hyperplane
            std::vector<T> w1(w.data(), w.data() + w.size());
            this->VIhandler->valueIteration(w1);

            std::vector<T> r1 = this->VIhandler->getResults();
            // only need the first m elements
            r1.resize(m);
            r = VectorMap<T>(r1.data(), r1.size());

            Phi.push_back(r);
            W.push_back(w);

            if (Phi.size() == 1) {
                vPrv = r;
            } else {
                vPrv = vt;
            }

            if (W.size() == 1 || w.dot(r) < w.dot(vb)) {
                vb = vt;
                this->secondaryOptimizer->minimize(vb, Phi, W);
            }

            tol = std::abs(this->fn->value(vt) - this->fn->value(vb));
            if (tol < eps) {
                std::cout << "loop exit due to small distance on threshold (tolerance: " << tol << ")\n";
                ++iter;
                break;
            }
            ++iter;
        }

        this->VIhandler->exit();

        {
            Vector<T> vOut = (vb + vt) * static_cast<T>(0.5);
            std::cout << "----------------------------------------------\n"
                      << "CUDA CONVEX QUERY terminates after " << iter << " iteration(s)\n"
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
    class ConvexQuery<double, int>;
}
