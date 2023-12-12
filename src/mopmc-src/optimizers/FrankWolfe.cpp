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

        if (this->fwOptMethod == LINOPT) {
            point = argminByLinOpt(Vertices, initialPoint, PolytopeType::Vertex, this->lineSearch);
        }
        if (this->fwOptMethod == AWAY_STEP) {
            point = argminByAwayStep(Vertices, initialPoint, this->lineSearch);
        }
        if (this->fwOptMethod == BLENDED) {
            point = argminByBlendedGD(Vertices, initialPoint, this->lineSearch);
        }

        return 0;
    }

    template<typename V>
    Vector<V> FrankWolfe<V>::argminByBlendedGD(const std::vector<Vector<V>> &Phi,
                                               Vector<V> &xIn,
                                               bool doLineSearch) {
        if (Phi.empty())
            throw std::runtime_error("The set of vertices cannot be empty");

        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearch<V> lineSearch(this->fn);
        auto m = Phi[0].size();
        Vector<V> xCurrent(m), xNew(m), vStar(m);

        const V epsilon{1.e-12}, gamma0{static_cast<V>(0.1)};
        const uint64_t maxIter = 1e4;
        V gamma, gammaMax, tol0, tol1;
        bool isFw;
        uint64_t t;
        uint64_t k = Phi.size();
        this->alpha(k-1) = static_cast<V>(0.);

        if (Phi.size() == 1) {
            this->alpha(0) = static_cast<V>(1.);
            this->activeSet.insert(0);
        }

        //estimate initial gap


        xNew.setZero();
        for (uint_fast64_t i = 0; i < Phi.size(); ++i) {
            assert(xNew.size() == Phi[i].size());
            xNew += this->alpha(i) * Phi[i];
        }

        V delta = std::numeric_limits<V>::min();
        for (uint_fast64_t i = 0; i < k; ++i) {
            V c = (this->fn->gradient(xNew)).dot(xNew - Phi[i]) / 2;
            if (delta < c) {
                delta = c;
            }
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
            for (auto j : this->activeSet) {
                assert(Phi[j].size() == this->fn->subgradient(xCurrent).size());
                if (Phi[j].dot(this->fn->subgradient(xCurrent)) > dec){
                    awId = j;
                }
            }
            Vector<V> awVec = xCurrent - Phi[awId];

            tol0 = static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(Phi[fwId] - xCurrent);
            tol1 = static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(xCurrent - Phi[awId]);

            if (tol0 <= epsilon) {
                break;
            }

            if (tol0 + tol1 >= delta) {
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
                        this->activeSet.clear();
                        this->activeSet.insert(fwId);
                    } else {
                        this->activeSet.insert(fwId);
                    }

                    for (uint_fast64_t l = 0; l < Phi.size(); ++l) {
                        if (l != fwId) {
                            this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                        }
                    }
                    this->alpha(fwId) = (static_cast<V>(1.) - gamma) * this->alpha(fwId) + gamma;
                } else {
                    if (gamma == gammaMax) {
                        this->activeSet.erase(awId);
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
            else {
                vStar = xCurrent - this->fn->subgradient(xCurrent) * delta * static_cast<V>(0.8);
                bool feasible;
                linOpt.checkPointInConvexHull(Phi, vStar, feasible);
                if (feasible) {
                    gamma = lineSearch.findOptimalDecentDistance(xCurrent, vStar, static_cast<V>(1.));
                    xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * vStar;
                } else {
                    delta *= static_cast<V>(0.5);
                }
            }
        }

        //std::cout << "*Frank-Wolfe* stops at iteration " << t << ", tolerance: " << std::max(tol0,tol1) << " \n";

        return xNew;
    };

    //Frank-Wolfe with away steps
    template<typename V>
    Vector<V> FrankWolfe<V>::argminByAwayStep(const std::vector<Vector<V>> &Phi,
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
        this->alpha(k-1) = static_cast<V>(0.);

        if (Phi.size() == 1) {
            this->alpha(0) = static_cast<V>(1.);
            this->activeSet.insert(0);
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
            for (auto j : this->activeSet) {
                assert(Phi[j].size() == this->fn->subgradient(xCurrent).size());
                if (Phi[j].dot(this->fn->subgradient(xCurrent)) > dec){
                    awId = j;
                }
            }
            Vector<V> awVec = xCurrent - Phi[awId];

            tol0 = static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(Phi[fwId] - xCurrent);
            tol1 = static_cast<V>(-1.) * this->fn->subgradient(xCurrent).dot(xCurrent - Phi[awId]);
            if (tol0 <= epsilon) {
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
                    this->activeSet.clear();
                    this->activeSet.insert(fwId);
                } else {
                    this->activeSet.insert(fwId);
                }

                for (uint_fast64_t l = 0; l < Phi.size(); ++l) {
                    if (l != fwId) {
                        this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                    }
                }
                this->alpha(fwId) = (static_cast<V>(1.) - gamma) * this->alpha(fwId) + gamma;
            } else {
                if (gamma == gammaMax) {
                    this->activeSet.erase(awId);
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
    Vector<V> FrankWolfe<V>::argminByLinOpt(const std::vector<Vector<V>> &Phi, Vector<V> &xIn, PolytopeType rep, bool doLineSearch) {

        std::vector<Vector<V>> W0;
        return this->argminByLinOpt(Phi, W0, xIn, rep, doLineSearch);
    }


    template<typename V>
    FrankWolfe<V>::FrankWolfe(FWOptMethod optMethod, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
            : fwOptMethod(optMethod), BaseOptimizer<V>(f) {}


    template<typename V>
    FrankWolfe<V>::FrankWolfe(convex_functions::BaseConvexFunction<V> *f, uint64_t maxSize)
            : BaseOptimizer<V>(f) { alpha.resize(maxSize); }

    template
    class FrankWolfe<double>;
}
