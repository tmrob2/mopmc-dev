//
// Created by guoxin on 4/12/23.
//

#include "ProjectedGradientDescent.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace mopmc::optimization::optimizers {

    /**
    * Argsort(for descending sort)
    * @param V array element type
    * @param vec input array
    * @return indices w.r.t sorted array
    */
    template<typename V>
    std::vector<size_t> argsort(const Vector<V> &vec) {
        std::vector<size_t> indices(vec.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&vec](int left, int right) -> bool {
                      // sort indices according to corresponding array element
                      return vec(left) > vec(right);
                  });

        return indices;
    }

    template<typename V>
    ProjectedGradientDescent<V>::ProjectedGradientDescent(convex_functions::BaseConvexFunction<V> *f) : fn(f) {}

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::argmin(Vector<V> &iniPoint,
                                                  std::vector<Vector<V>> &Phi,
                                                  std::vector<Vector<V>> &W) {

        uint64_t maxIter = 1000;
        uint64_t m = iniPoint.size();
        V gamma = static_cast<V>(0.1);
        V epsilon = static_cast<V>(1.e-8);
        Vector<V> xCurrent = iniPoint, xNew(m), xTemp(m);
        Vector<V> grad(m);
        uint_fast64_t it;
        for (it = 0; it < maxIter; ++it) {
            grad = this->fn->subgradient(xCurrent);
            xTemp = xCurrent - gamma * grad; //it + static_cast<V>(1.))
            xNew = projectToNearestHyperplane(xTemp, Phi, W);
            V error = (xNew - xCurrent).template lpNorm<1>();
            if (error < epsilon) {
                xCurrent = xNew;
                break;
            }
            xCurrent = xNew;
        }
        std::cout << "*Projected GD* stops at iteration " << it << "\n";
        return xCurrent;
    }

    //todo this function is not correct for now.
    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::argminUnitSimplexProjection(Vector<V> &iniPoint,
                                                                       std::vector<Vector<V>> &Phi) {
        uint64_t maxIter = 1000;
        uint64_t m = iniPoint.size();
        V gamma = static_cast<V>(0.1);
        V epsilon = static_cast<V>(1.e-8);
        Vector<V> xCurrent = iniPoint, xNew(m), xTemp(m);
        Vector<V> grad(m);
        uint_fast64_t it;
        for (it = 0; it < maxIter; ++it) {
            grad = this->fn->subgradient(xCurrent);
            xTemp = xCurrent - gamma * grad;
            xNew = projectToUnitSimplex(xTemp);
            V error = (xNew - xCurrent).template lpNorm<1>();
            if (error < epsilon) {
                xCurrent = xNew;
                break;
            }
            xCurrent = xNew;
        }
        std::cout << "*Projected GD* (to simplex) stops at iteration " << it << "\n";
        return xCurrent;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::projectToNearestHyperplane(Vector<V> &x,
                                                                      std::vector<Vector<V>> &Phi,
                                                                      std::vector<Vector<V>> &W) {
        assert(W.size() == Phi.size());
        assert(!Phi.empty());
        uint64_t m = Phi[0].size();
        V shortest = std::numeric_limits<V>::max();
        Vector<V> xProj = x;
        for (uint_fast64_t i = 0; i < Phi.size(); ++i) {
            V e = Phi[i].template lpNorm<2>();
            V distance = W[i].dot(x - Phi[i]) / std::pow(e, 2);
            if (distance > 0 && distance < shortest) {
                shortest = distance;
                xProj = x - (shortest * W[i]);
            }
        }
        return xProj;
    }

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::projectToUnitSimplex(Vector<V> &x) {
        assert(x.size() > 0);
        uint64_t m = x.size();
        //It seems that sorting is not well supported
        // in the Storm's Eigen version.
        // We convert the vector to a standard vector.
        std::vector<V> y(x.data(), x.data() + m);
        std::vector<uint64_t> ids = argsort(x);

        V tmpsum = static_cast<V>(0.), tmax;
        bool bget = false;
        for (uint_fast64_t i = 0; i < m - 1; ++i) {
            tmpsum += y[ids[i]];
            tmax = (tmpsum - static_cast<V>(1.)) / static_cast<V>(i);
            if (tmax >= y[ids[i + 1]]) {
                bget = true;
                break;
            }
        }

        if (!bget) {
            tmax = (tmpsum + y[ids[m - 1]] - static_cast<V>(1.)) / static_cast<V>(m);
        }

        for (uint_fast64_t j = 0; j < m; ++j) {
            y[ids[j]] = std::max(y[ids[j]] - tmax, static_cast<V>(0.));
        }
        Vector<V> xProj = VectorMap<V>(y.data(), m);
        std::cout << xProj << "\n" << "---\n";

        return xProj;
    }

    template
    class ProjectedGradientDescent<double>;
}
