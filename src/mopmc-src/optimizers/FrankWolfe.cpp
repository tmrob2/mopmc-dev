//
// Created by guoxin on 24/11/23.
//

#include <cmath>
#include <iostream>
#include "FrankWolfe.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    V cosine(Vector<V> x, Vector<V> y, V c) {
        V a = x.template lpNorm<2>();
        V b = y.template lpNorm<2>();
        if (a == 0 || b == 0) {
            return c;
        } else {
            return x.dot(y) / (a * b);
        }
    }

    //Frank-Wolfe with away steps
    template<typename V>
    Vector<V> FrankWolfe<V>::argminByAwayStep(std::vector<Vector<V>> &Phi,
                                              Vector<V> &xIn,
                                              bool doLineSearch){
        if (Phi.empty())
            throw std::runtime_error("The set of vertices cannot be empty");

        mopmc::optimization::optimizers::LineSearch<V> lineSearch(this->fn);
        auto m = Phi[0].size();
        Vector<V> xCurrent(m), xNew(m), vStar(m);

        const V epsilon{1.e-12}, gamma0{static_cast<V>(0.1)};
        const uint64_t maxIter = 1e4;
        V gamma, gammaMax, tol0, tol1;
        bool isFw;
        uint64_t t;
        uint64_t k = Phi.size();
        this->alpha(k-1) = 0;

        if (Phi.size() == 1) {
            this->alpha(0) = static_cast<V>(1.);
            this->S.insert(0);
        }

        xNew.setZero();
        for (uint_fast64_t i = 0; i < Phi.size(); ++i) {
            assert(xNew.size() == Phi[i].size());
            xNew += this->alpha(i) * Phi[i];
        }

        for (t = 1; t < maxIter; ++t) {
            xCurrent = xNew;
            Vector<V> d = this->fn->subgradient(xCurrent);

            uint64_t fwId = 0;
            V inc = std::numeric_limits<V>::max();
            for (uint_fast64_t i = 0; i < Phi.size(); ++i) {
                if (Phi[i].dot(this->fn->subgradient(xCurrent)) < inc){
                    fwId = i;
                }
            }
            Vector<V> fwVec = (Phi[fwId] - xCurrent);

            uint64_t awId = 0;
            V dec = std::numeric_limits<V>::min();
            for (auto j : S) {
                assert(Phi[j].size() == this->fn->subgradient(xCurrent).size());
                if (Phi[j].dot(this->fn->subgradient(xCurrent)) > dec){
                    awId = j;
                }
            }
            Vector<V> awVec = xCurrent - Phi[awId];

            tol0 = static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(Phi[fwId] - xCurrent);
            tol1 = static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(xCurrent - Phi[awId]);
            if (tol0 <= epsilon && tol1 < epsilon) {
                break;
            }

            if (static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(fwVec - awVec) >= 0.){
                isFw = true;
                vStar = xCurrent + fwVec;
                gammaMax = static_cast<V>(1.);
            } else {
                isFw = false;
                vStar = xCurrent + awVec;
                gammaMax = this->alpha(awId) / (static_cast<V>(1.) - this->alpha(awId));
            }

            if (doLineSearch) {
                gamma = lineSearch.findOptimalDecentDistance(xCurrent, vStar, gammaMax);
            } else {
                gamma = gamma0 * static_cast<V>(2) / static_cast<V>(t + 2);
            }

            if (isFw) {
                if (gamma == gammaMax) {
                    this->S.clear();
                    this->S.insert(fwId);
                } else {
                    this->S.insert(fwId);
                }

                for (uint_fast64_t l = 0; l < Phi.size(); ++l) {
                    if (l != fwId) {
                        this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                    }
                }
                this->alpha(fwId) = (static_cast<V>(1.) - gamma) * this->alpha(fwId) + gamma;
            } else {
                if (gamma == gammaMax) {
                    this->S.erase(awId);
                }
                for (uint_fast64_t l = 0; l < Phi.size(); ++l) {
                    if (l != awId) {
                        this->alpha(l) = (static_cast<V>(1.) + gamma) * this->alpha(l);
                    }
                }
                this->alpha(awId) = (static_cast<V>(1.) + gamma) * this->alpha(awId) - gamma;
            }

            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * vStar;
        }

        std::cout << "*Frank-Wolfe* stops at iteration " << t << ", tolerance: " << std::max(tol0,tol1) << " \n";

        return xNew;
    }

    //Frank-Wolfe with LinOpt (LP)
    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(std::vector<Vector<V>> &Phi,
                                    std::vector<Vector<V>> &W,
                                    Vector<V> &xIn,
                                    PolytopeType polytopeType,
                                    bool doLineSearch) {
        if (Phi.empty()) {
            throw std::runtime_error("The set of vertices cannot be empty");
        }
        if (polytopeType == Halfspace) {
            if (W.size() != Phi.size()) {
                throw std::runtime_error("The numbers of vertices and weights are not the same");
            }
        }

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearch<V> lineSearch(this->fn);

        auto m = xIn.size();
        Vector<V> xCurrent(m), xNew = xIn, vStar(m);
        const V epsilon{1.e-5}, gamma0{static_cast<V>(0.1)};
        const uint64_t maxIter = 10000;
        V gamma, tolerance;
        uint64_t i;

        for (i = 0; i < maxIter; ++i) {

            xCurrent = xNew;
            Vector<V> d = this->fn->subgradient(xCurrent);
            if (polytopeType == Vertex) {
                linOpt.optimizeVtx(Phi, polytopeType, d, vStar);
            }
            if (polytopeType == Halfspace) {
                linOpt.optimizeHlsp(Phi, W, polytopeType, d, vStar);
            }

            tolerance = static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(vStar - xCurrent);
            if (tolerance <= epsilon) {
                break;
            }
            if (doLineSearch) {
                gamma = lineSearch.findOptimalDecentDistance(xCurrent, vStar);
            } else {
                gamma = gamma0 * static_cast<V>(2) / static_cast<V>(i + 2);
            }
            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * vStar;
        }
        std::cout << "*Frank-Wolfe* stops at iteration " << i << ", tolerance: " << tolerance << " \n";
        return xNew;
    }


    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(std::vector<Vector<V>> &Phi, Vector<V> &xIn, PolytopeType rep, bool doLineSearch) {

        std::vector<Vector<V>> W0;
        return this->argmin(Phi, W0, xIn, rep, doLineSearch);
    }

    template<typename V>
    FrankWolfe<V>::FrankWolfe(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
            : fn(f) {}

    template<typename V>
    FrankWolfe<V>::FrankWolfe(convex_functions::BaseConvexFunction<V> *f, uint64_t maxSize)
            : fn(f) { alpha.resize(maxSize); }

    template
    class FrankWolfe<double>;
}
