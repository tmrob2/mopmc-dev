//
// Created by guoxin on 4/12/23.
//

#include "ProjectedGradientDescent.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace mopmc::optimization::optimizers{

    template<typename V>
    ProjectedGradientDescent<V>::ProjectedGradientDescent(convex_functions::BaseConvexFunction<V> *f) : fn(f){}

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::argmin(Vector<V> &iniPoint,
                                                  std::vector<Vector<V>> &Phi,
                                                  std::vector<Vector<V>> &W) {

        uint64_t maxIter = 1000;
        uint64_t m = iniPoint.size();
        V gamma = static_cast<V>(0.1);
        V epsilon = static_cast<V>(1.e-8);
        Vector<V> xCurrent=iniPoint, xNew(m), xTemp(m);
        Vector<V> grad(m);
        assert(xTemp.size() == m);
        uint_fast64_t it;
        for (it = 0; it < maxIter; ++it) {
            grad = this->fn->subgradient(xCurrent);
            xTemp = xCurrent - gamma * grad; //it + static_cast<V>(1.))
            xNew = findNearestProjectedPoint(xTemp, Phi, W);
            V error = (xNew - xCurrent).template lpNorm<1>();
            if (error < epsilon) {
                xCurrent = xNew;
                break;
            }
            xCurrent = xNew;
        }
        std::cout << "*Projected GD* stops at iteration " << it <<"\n";
        return xCurrent;
    };

    template<typename V>
    Vector<V> ProjectedGradientDescent<V>::findNearestProjectedPoint(Vector<V> &x,
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

    template class ProjectedGradientDescent<double>;
}
