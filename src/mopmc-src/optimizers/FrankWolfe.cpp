//
// Created by guoxin on 24/11/23.
//

#include "FrankWolfe.h"
#include "../convex-functions/auxiliary/Lincom.h"
#include <cmath>
#include <iostream>

namespace mopmc::optimization::optimizers {

    template<typename V>
    V cosine(const Vector<V> &x, const Vector<V> &y, const V &c) {
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
        point = argmin(Vertices);
        return 0;
    }

    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(const std::vector<Vector<V>> &Vertices) {

        initialize(Vertices);
        uint64_t t = 0;
        while (t < maxIter) {
            xCurrent = xNew;
            dXCurrent = this->fn->subgradient(xCurrent);
            computeForwardStepIndexAndVector(Vertices);
            computeAwayStepIndexAndVector(Vertices);
            epsFwd = static_cast<V>(-1.) * dXCurrent.dot(Vertices[fwdInd] - xCurrent);
            epsAwy = static_cast<V>(-1.) * dXCurrent.dot(xCurrent - Vertices[awyInd]);
            if (epsFwd <= tolerance) {
                std::cout << "FW loop exits due to small tolerance: " << epsFwd << "\n";
                ++t;
                break;
            }
            if (cosine(fwdVec, dXCurrent, 0.) > toleranceCosine) {
                std::cout << "FW loop exits due to small cosine: " << cosine(fwdVec, dXCurrent, 0.) << "\n";
                ++t;
                break;
            }

            switch (this->fwOption) {
                case LINOPT: {
                    PolytopeType polytopeType = PolytopeType::Vertex;
                    this->linOpt.optimizeVtx(Vertices, polytopeType, dXCurrent, xNewEx);
                    gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx);
                    xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
                    break;
                }
                case AWAY_STEP: {
                    updateWithForwardOrAwayStep();
                    break;
                }
                case BLENDED: {
                    if (epsFwd + epsAwy >= delta) {
                        updateWithForwardOrAwayStep();
                    } else {
                        int feasible = -1;
                        this->linOpt.checkPointInConvexHull(Vertices, (xCurrent - dXCurrent * delta), feasible);
                        if (feasible == 0) {
                            xNewEx = xCurrent - dXCurrent * delta;
                            gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, static_cast<V>(1.));
                            xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
                        } else if (feasible == 2) {
                            delta *= static_cast<V>(0.5);
                        } else {
                            printf("[Warning] ret = %i\n", feasible);
                            ++t;
                            break;
                            //throw std::runtime_error("linopt error");
                        }
                    }
                    break;
                }
                case BLENDED_STEP_OPT: {
                    this->linOpt.findMaximumFeasibleStep(Vertices, dXCurrent, xCurrent, stepSize);
                    if (stepSize > delta * scale2) {
                        xNewEx = xCurrent - dXCurrent * stepSize;
                        gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, static_cast<V>(1.));
                        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
                    } else {
                        delta *= static_cast<V>(0.5);
                    }
                    break;
                }
            }
            ++t;
        }
        std::cout << "Frank-Wolfe stops at iteration: " << t << "\n";
        return xNew;
    }

    template<typename V>
    void FrankWolfe<V>::initialize(const std::vector<Vector<V>> &Vertices) {
        if (Vertices.empty())
            throw std::runtime_error("The set of vertices cannot be empty");

        size = Vertices.size();
        dimension = Vertices[0].size();
        xCurrent.resize(dimension);
        xNew.resize(dimension);
        xNewEx.resize(dimension);

        this->alpha.conservativeResize(size);
        this->alpha(size - 1) = static_cast<V>(0.);

        if (size == 1) {
            this->alpha(0) = static_cast<V>(1.);
            this->activeSet.insert(0);
        }

        xNew.setZero();
        for (uint_fast64_t i = 0; i < size; ++i) {
            assert(xNew.size() == Vertices[i].size());
            xNew += this->alpha(i) * Vertices[i];
        }

        //estimate initial gap
        delta = std::numeric_limits<V>::min();
        for (uint_fast64_t i = 0; i < size; ++i) {
            const V c = (this->fn->gradient(xNew)).dot(xNew - Vertices[i]) / scale1;
            if (c > delta) {
                delta = c;
            }
        }
    }

    template<typename V>
    void FrankWolfe<V>::updateWithSimplexGradientDescent(const std::vector<Vector<V>> &Vertices) {
        uint64_t n = activeSet.size();
        Vector<V> c = Vector<V>::Zero(n);
        for (auto i: activeSet) {
            c(i) += dXCurrent.dot(Vertices[i]);
        }
        Vector<V> cProj = c - (c.sum() / this->size) * c;
        if (cProj.isZero()) {
            const auto ind_ptr = activeSet.begin();
            activeSet.clear();
            activeSet.insert(*ind_ptr);
            alpha.setZero();
            alpha(*ind_ptr) = static_cast<V>(1.);
            xNewEx = Vertices[*ind_ptr];
            xNew = xNewEx;
        } else {
            V e = 0.;
            uint64_t indMax = n;
            for (auto i: activeSet) {
                if (cProj(i) > 0) {
                    if (alpha(i) / cProj(i) < e) {
                        e = alpha(i) / cProj(i);
                        indMax = i;
                    }
                }
            }
            xNewEx = xCurrent;
            for (auto i: activeSet) {
                xNewEx -= e * cProj(i) * Vertices[i];
            }
            if (this->fn->value(xCurrent) < this->fn->value(xNewEx)) {
                xNew = xNewEx;
                activeSet.erase(indMax);
                gamma = 1.0;
            } else {
                gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, 1.0);
                xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
            }
            for (uint_fast64_t i = 0; i < this->size; ++i) {
                if (activeSet.count(i)) {
                    alpha(i) = (1 - gamma) * alpha(i) + gamma * (alpha(i) - e * cProj(i));
                } else {
                    alpha(i) = 0. ;
                }
            }
            const V p = alpha.sum();
            for (auto i: activeSet) {
                alpha(i) /= p;
            }
        }
    }

    template<typename V>
    void FrankWolfe<V>::updateWithForwardOrAwayStep() {
        if (static_cast<V>(-1.) * dXCurrent.dot(fwdVec - awyVec) >= 0.) {
            isFwd = true;
            xNewEx = xCurrent + fwdVec;
            gammaMax = static_cast<V>(1.);
        } else {
            isFwd = false;
            xNewEx = xCurrent + awyVec;
            gammaMax = this->alpha(awyInd) / (static_cast<V>(1.) - this->alpha(awyInd));
        }

        gamma = this->lineSearcher.findOptimalDecentDistance(xCurrent, xNewEx, gammaMax);

        if (isFwd) {
            if (gamma == gammaMax) {
                this->activeSet.clear();
                this->activeSet.insert(fwdInd);
            } else {
                this->activeSet.insert(fwdInd);
            }

            for (uint_fast64_t l = 0; l < this->size; ++l) {
                if (l != fwdInd) {
                    this->alpha(l) = (static_cast<V>(1.) - gamma) * this->alpha(l);
                }
            }
            this->alpha(fwdInd) = (static_cast<V>(1.) - gamma) * this->alpha(fwdInd) + gamma;
        } else {
            if (gamma == gammaMax) {
                this->activeSet.erase(awyInd);
            }
            for (uint_fast64_t l = 0; l < this->size; ++l) {
                if (l != awyInd) {
                    this->alpha(l) = (static_cast<V>(1.) + gamma) * this->alpha(l);
                }
            }
            this->alpha(awyInd) = (static_cast<V>(1.) + gamma) * this->alpha(awyInd) - gamma;
        }
        xNew = (static_cast<V>(1.) - gamma) * xCurrent + gamma * xNewEx;
    }

    template<typename V>
    void FrankWolfe<V>::computeAwayStepIndexAndVector(const std::vector<Vector<V>> &Vertices) {
        awyInd = 0;
        V inc = std::numeric_limits<V>::min();
        for (auto j: this->activeSet) {
            //assert(Vertices[j].size() == dXCurrent.size());
            if (Vertices[j].dot(dXCurrent) > inc) {
                inc = Vertices[j].dot(dXCurrent);
                awyInd = j;
            }
        }
        awyVec = xCurrent - Vertices[awyInd];
    }

    template<typename V>
    void FrankWolfe<V>::computeForwardStepIndexAndVector(const std::vector<Vector<V>> &Vertices) {
        fwdInd = 0;
        V dec = std::numeric_limits<V>::max();
        for (uint_fast64_t i = 0; i < Vertices.size(); ++i) {
            if (Vertices[i].dot(dXCurrent) < dec) {
                dec = Vertices[i].dot(dXCurrent);
                fwdInd = i;
            }
        }
        fwdVec = (Vertices[fwdInd] - xCurrent);
    }

    template<typename V>
    FrankWolfe<V>::FrankWolfe(FWOption option, mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
        : fwOption(option), BaseOptimizer<V>(f) {
        this->lineSearcher = mopmc::optimization::optimizers::LineSearcher<V>(f);
    }

    template class FrankWolfe<double>;
}// namespace mopmc::optimization::optimizers
