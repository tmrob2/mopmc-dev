//
// Created by guoxin on 24/11/23.
//

#include <cmath>
#include <iostream>
#include "FrankWolfe.h"
#include "../convex-functions/auxiliary/Lincom.h"

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

    template<typename V>
    int FrankWolfe<V>::minimize(Vector<V> &point, const std::vector<Vector<V>> &Vertices) {
        Vector<V> initialPoint = point;

        if (this->fwOption == LINOPT) {
            point = argminByLinOpt(Vertices, initialPoint, PolytopeType::Vertex, this->lineSearch);
        }
        if (this->fwOption == AWAY_STEP) {
            point = argminWithAwayStep(Vertices, this->lineSearch);
        }
        if (this->fwOption == BLENDED) {
            bool feasibilityCheckOnly = true;
            point = argminByBlendedGD(Vertices, this->lineSearch, feasibilityCheckOnly);
        }
        if (this->fwOption == BLENDED_STEP_OPT) {
            bool feasibilityCheckOnly = false;
            point = argminByBlendedGD(Vertices, this->lineSearch, feasibilityCheckOnly);
        }
        return 0;
    }

    template<typename V>
    Vector<V> FrankWolfe<V>::argminByBlendedGD(const std::vector<Vector<V>> &Vertices,
                                               bool doLineSearch, bool feasibilityCheckOnly) {
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearch<V> lineSearch(this->fn);
        const uint64_t k = Vertices.size();
        const uint64_t m = Vertices[0].size();
        Vector<V> xCurrent(m), xNew(m), xTemp(m);
        const V epsilon{1.e-12}, gamma0{static_cast<V>(0.1)}, impr{static_cast<V>(0.5)};
        const uint64_t maxIter = 1e3;
        V gamma, gammaMax, tolFw, tolAw, stepSize, delta;
        bool isFw;

        this->alpha.conservativeResize(k);
        this->alpha(k-1) = static_cast<V>(0.);

        if (k == 1) {
            this->alpha(0) = static_cast<V>(1.);
            this->activeSet.insert(0);
        }

        //estimate initial gap
        xNew.setZero();
        for (uint_fast64_t i = 0; i < k; ++i) {
            assert(xNew.size() == Vertices[i].size());
            xNew += this->alpha(i) * Vertices[i];
        }

        delta = std::numeric_limits<V>::min();
        for (uint_fast64_t i = 0; i < k; ++i) {
            const V c = (this->fn->gradient(xNew)).dot(xNew - Vertices[i]) * 2.;
            if (c > delta) {
                delta = c;
            }
        }

        uint64_t t;
        for (t = 0; t < maxIter; ++t) {
            xCurrent = xNew;
            Vector<V> dXCurrent = this->fn->subgradient(xCurrent);
            uint64_t fwId = 0;
            V dec = std::numeric_limits<V>::max();
            for (uint_fast64_t i = 0; i < k; ++i) {
                if (Vertices[i].dot(dXCurrent) < dec){
                    dec = Vertices[i].dot(dXCurrent);
                    fwId = i;
                }
            }
            Vector<V> fwVec = (Vertices[fwId] - xCurrent);

            uint64_t awId = 0;
            V inc = std::numeric_limits<V>::min();
            for (auto j : this->activeSet) {
                assert(Vertices[j].size() == dXCurrent.size());
                if (Vertices[j].dot(dXCurrent) > inc){
                    inc = Vertices[j].dot(dXCurrent);
                    awId = j;
                }
            }
            Vector<V> awVec = xCurrent - Vertices[awId];

            tolFw = static_cast<V>(-1.) * dXCurrent.dot(Vertices[fwId] - xCurrent);
            tolAw = static_cast<V>(-1.) * dXCurrent.dot(xCurrent - Vertices[awId]);
            if (tolFw <= epsilon) {
                std::cout << "FW loop breaks due to small tol: " << tolFw << "\n";
                break;
            }

            if (tolFw + tolAw >= delta) {
                if (static_cast<V>(-1.) * dXCurrent.dot(fwVec - awVec) >= 0.){
                    isFw = true;
                    xTemp = xCurrent + fwVec;
                    gammaMax = static_cast<V>(1.);
                } else {
                    isFw = false;
                    xTemp = xCurrent + awVec;
                    gammaMax = this->alpha(awId) / (static_cast<V>(1.) - this->alpha(awId));
                }

                if (doLineSearch) {
                    gamma = lineSearch.findOptimalDecentDistance(xCurrent, xTemp, gammaMax);
                } else {
                    gamma = gamma0 * static_cast<V>(2) / static_cast<V>(t + 2);
                }

                if (isFw) {
                    if (gamma == gammaMax) {
                        this->activeSet.clear();
                        this->activeSet.insert(fwId);
                    } else {
                        this->activeSet.insert(fwId);
                    }

                    for (uint_fast64_t l = 0; l < k; ++l) {
                        if (l != fwId) {
                            this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                        }
                    }
                    this->alpha(fwId) = (static_cast<V>(1.) - gamma) * this->alpha(fwId) + gamma;
                } else {
                    if (gamma == gammaMax) {
                        this->activeSet.erase(awId);
                    }
                    for (uint_fast64_t l = 0; l < k; ++l) {
                        if (l != awId) {
                            this->alpha(l) = (static_cast<V>(1.) + gamma) * this->alpha(l);
                        }
                    }
                    this->alpha(awId) = (static_cast<V>(1.) + gamma) * this->alpha(awId) - gamma;
                }
                xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xTemp;
            }
            else {
                //std::cout << "alpha: " << this->alpha << "\n" << "xCurrent: " << xCurrent <<"\n";
                if (feasibilityCheckOnly) {
                    int feasible = -1;
                    linOpt.checkPointInConvexHull(Vertices, (xCurrent - dXCurrent * delta), feasible);
                    if (feasible == 0) {
                        xTemp = xCurrent - dXCurrent * delta;
                        gamma = lineSearch.findOptimalDecentDistance(xCurrent, xTemp, static_cast<V>(1.));
                        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xTemp;
                    } else if (feasible == 2) {
                        delta *= static_cast<V>(0.5);
                    } else {
                        printf("[Warning] ret = %i\n", feasible);
                        break;
                        //throw std::runtime_error("linopt error");
                        }
                } else {
                    linOpt.findMaximumFeasibleStep(Vertices, dXCurrent, xCurrent, stepSize);
                    if (stepSize > delta * impr) {
                        xTemp = xCurrent - dXCurrent * stepSize;
                        gamma = lineSearch.findOptimalDecentDistance(xCurrent, xTemp, static_cast<V>(1.));
                        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xTemp;
                    } else {
                        delta *= static_cast<V>(0.5);
                    }
                }
            }
        }
        std::cout << "*Blended GD* stops at iteration: " << t << ", delta: " << delta << " \n";
        return xNew;
    };

    //Frank-Wolfe with away steps
    template<typename V>
    Vector<V> FrankWolfe<V>::argminWithAwayStep(const std::vector<Vector<V>> &Vertices,
                                                bool doLineSearch){
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");

        mopmc::optimization::optimizers::LineSearch<V> lineSearch(this->fn);
        const uint64_t m = Vertices[0].size();
        const uint64_t k = Vertices.size();
        Vector<V> xCurrent(m), xNew(m), xTemp(m);
        const V epsilon{1.e-12}, gamma0{static_cast<V>(0.1)};
        const uint64_t maxIter = 2e4;
        V gamma, gammaMax, tolFw, tolAw;
        bool isFw;
        this->alpha.conservativeResize(k);
        this->alpha(k-1) = static_cast<V>(0.);

        if (Vertices.size() == 1) {
            this->alpha(0) = static_cast<V>(1.);
            this->activeSet.insert(0);
        }

        xNew.setZero();
        for (uint_fast64_t i = 0; i < k; ++i) {
            assert(xNew.size() == Vertices[i].size());
            xNew += this->alpha(i) * Vertices[i];
        }

        uint64_t t;
        for (t = 1; t < maxIter; ++t) {
            xCurrent = xNew;
            Vector<V> dXCurrent = this->fn->subgradient(xCurrent);

            uint64_t fwId = 0;
            V dec = std::numeric_limits<V>::max();
            for (uint_fast64_t i = 0; i < k; ++i) {
                if (Vertices[i].dot(dXCurrent) < dec){
                    dec = Vertices[i].dot(xCurrent);
                    fwId = i;
                }
            }
            Vector<V> fwVec = (Vertices[fwId] - xCurrent);

            uint64_t awId = 0;
            V inc = std::numeric_limits<V>::min();
            for (auto j : this->activeSet) {
                assert(Vertices[j].size() == dXCurrent.size());
                if (Vertices[j].dot(dXCurrent) > inc){
                    inc = Vertices[j].dot(xCurrent);
                    awId = j;
                }
            }
            Vector<V> awVec = xCurrent - Vertices[awId];

            tolFw = static_cast<V>(-1.) * dXCurrent.dot(Vertices[fwId] - xCurrent);
            tolAw = static_cast<V>(-1.) * dXCurrent.dot(xCurrent - Vertices[awId]);
            if (tolFw <= epsilon) {
                break;
            }

            if (static_cast<V>(-1.) * dXCurrent.dot(fwVec - awVec) >= 0.){
                isFw = true;
                xTemp = xCurrent + fwVec;
                gammaMax = static_cast<V>(1.);
            } else {
                isFw = false;
                xTemp = xCurrent + awVec;
                gammaMax = this->alpha(awId) / (static_cast<V>(1.) - this->alpha(awId));
            }

            if (doLineSearch) {
                gamma = lineSearch.findOptimalDecentDistance(xCurrent, xTemp, gammaMax);
            } else {
                gamma = gamma0 * static_cast<V>(2) / static_cast<V>(t + 2);
            }

            if (isFw) {
                if (gamma == gammaMax) {
                    this->activeSet.clear();
                    this->activeSet.insert(fwId);
                } else {
                    this->activeSet.insert(fwId);
                }

                for (uint_fast64_t l = 0; l < k; ++l) {
                    if (l != fwId) {
                        this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                    }
                }
                this->alpha(fwId) = (static_cast<V>(1.) - gamma) * this->alpha(fwId) + gamma;
            } else {
                if (gamma == gammaMax) {
                    this->activeSet.erase(awId);
                }
                for (uint_fast64_t l = 0; l < k; ++l) {
                    if (l != awId) {
                        this->alpha(l) = (static_cast<V>(1.) + gamma) * this->alpha(l);
                    }
                }
                this->alpha(awId) = (static_cast<V>(1.) + gamma) * this->alpha(awId) - gamma;
            }
            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xTemp;
        }
        std::cout << "*Frank-Wolfe* stops at iteration " << t << ", tolerance: " << std::max(tolFw, tolAw) << " \n";

        return xNew;
    }

    //Frank-Wolfe with LinOpt (LP)
    template<typename V>
    Vector<V> FrankWolfe<V>::argminByLinOpt(const std::vector<Vector<V>> &Phi,
                                            const std::vector<Vector<V>> &W,
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

        auto m = Phi[0].size();
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
    Vector<V> FrankWolfe<V>::argminByLinOpt(const std::vector<Vector<V>> &Phi, Vector<V> &xIn, PolytopeType rep, bool doLineSearch) {

        std::vector<Vector<V>> W0;
        return this->argminByLinOpt(Phi, W0, xIn, rep, doLineSearch);
    }


    template<typename V>
    FrankWolfe<V>::FrankWolfe(FWOption option, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
            : fwOption(option), BaseOptimizer<V>(f) {}


    template<typename V>
    FrankWolfe<V>::FrankWolfe(convex_functions::BaseConvexFunction<V> *f, uint64_t maxSize)
            : BaseOptimizer<V>(f) { alpha.resize(maxSize); }

    template
    class FrankWolfe<double>;
}
