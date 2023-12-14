//
// Created by guoxin on 3/11/23.
//


#include <iostream>
#include <Eigen/Dense>
#include "ConvexQuery.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void ConvexQuery<T, I>::query() {

        this->VIhandler->initialize();
        const uint64_t n_objs = this->data_.objectiveCount;
        assert(this->data_.rowGroupIndices.size() == this->data_.colCount + 1);
        Vector<T> threshold = Eigen::Map<Vector<T>> (this->data_.thresholds.data(), n_objs);

        std::vector<Vector<T>> Vertices, WeightVectors;
        Vector<T> innerPointCurrent(n_objs), innerPointNew(n_objs), outerPoint(n_objs);
        Vector<T> vertex(n_objs), weightVector(n_objs);
        weightVector.setConstant(static_cast<T>(1.0) / n_objs);

        const double eps{1.e-6}, eps1{1.e-8}, eps2{1.e-6};
        const uint_fast64_t maxIter{100};
        T tol, tol1, tol2;
        uint_fast64_t iter = 0;

        while (iter < maxIter) {
            std::cout << "Main loop: Iteration " << iter << "\n";
            if (!Vertices.empty()) {
                innerPointNew = innerPointCurrent;
                this->primaryOptimizer->minimize(innerPointNew, Vertices);

                if (Vertices.size() >= 2) {
                    //tol2 = (vPrv - vt).template lpNorm<1>();
                    tol2 = (innerPointCurrent - innerPointNew).template lpNorm<Eigen::Infinity>();
                    if (tol2 < eps2) {
                        std::cout << "loop exit due to small improvement on (estimated) nearest point (tolerance: "
                            << tol2 << ")\n";
                        ++iter;
                        break;
                    }
                }

                Vector<T> grad = this->fn->subgradient(innerPointNew);
                tol1 = grad.template lpNorm<1>();
                if (tol1 < eps1) {
                    std::cout << "loop exit due to small gradient (tolerance: " << tol1 << ")\n";
                    ++iter;
                    break;
                }
                weightVector = static_cast<T>(-1.) * grad / grad.template lpNorm<1>();
            }

            // compute a new supporting hyperplane
            std::vector<T> weightVec1(weightVector.data(), weightVector.data() + weightVector.size());
            this->VIhandler->valueIteration(weightVec1);

            std::vector<T> vertex1 = this->VIhandler->getResults();
            vertex = VectorMap<T>(vertex1.data(), n_objs);

            Vertices.push_back(vertex);
            WeightVectors.push_back(weightVector);

            if (Vertices.size() == 1) {
                innerPointCurrent = vertex;
            } else {
                innerPointCurrent = innerPointNew;
            }

            if (WeightVectors.size() == 1 || weightVector.dot(vertex) < weightVector.dot(outerPoint)) {
                outerPoint = innerPointCurrent;
                this->secondaryOptimizer->minimize(outerPoint, Vertices, WeightVectors);
            }
            tol = std::abs(this->fn->value(innerPointCurrent) - this->fn->value(outerPoint));
            if (tol < eps) {
                std::cout << "loop exit due to small distance on threshold (tolerance: " << tol << ")\n";
                ++iter;
                break;
            }
            ++iter;
        }

        this->VIhandler->exit();

        {
            Vector<T> vOut = (outerPoint + innerPointNew) * static_cast<T>(0.5);
            std::cout << "----------------------------------------------\n"
                      << "CUDA CONVEX QUERY terminates after " << iter << " iteration(s)\n"
                      << "Estimated nearest point to threshold : [";
            for (int i = 0; i < n_objs; ++i) {
                std::cout << innerPointNew(i) << " ";
            }
            std::cout << "]\n"
                    << "Approximate distance: " << this->fn->value(innerPointNew)
                      //<< "Approximate distance between " << fn.value(vb) << " and " << fn.value(vt)
                      << "\n----------------------------------------------\n";
        }
    }

    template
    class ConvexQuery<double, int>;
}
