//
// Created by guoxin on 24/11/23.
//

#include <cmath>
#include <iostream>
#include "FrankWolfe.h"
#include "../convex-functions/BaseConvexFunction.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(std::vector<Vector<V>> &Phi,
                                    std::vector<Vector<V>> &W,
                                    Vector<V> &xIn,
                                    PolytopeType rep,
                                    bool doLineSearch) {
        if (Phi.empty()) {
            throw std::runtime_error("The set of vertices cannot be empty");
        }
        if (rep == Halfspace) {
            if (W.size() != Phi.size()) {
                throw std::runtime_error("The numbers of vertices and weights are not the same");
            }
        }
        mopmc::optimization::optimizers::LinOpt<V> linOpt;
        mopmc::optimization::optimizers::LineSearch<V> lineSearch(this->fn);
        auto m = xIn.size();
        Vector<V> xOld(m), xNew = xIn;
        Vector<V> vStar(m);
        for (int i = 0; i < maxIter; ++i) {
            std::cout << "**xNew in FrankWolfe**: [" << xNew(0) <<", "<< xNew(1) << "]\n";
            xOld = xNew;
            Vector<V> d = this->fn->subgradient(xOld);
            if (rep == Vertex) {
                //std::cout << "**vStar for VRep before linOpt: [";
                for(int j=0; j < vStar.size(); ++j) {
                    std::cout << vStar(j) << " ";
                }
                std::cout <<"]\n";
                linOpt.argmin(Phi, rep, d, vStar);
                std::cout << "**vStar for VRep after linOpt after " << i << " iteration: [";
                for(int j=0; j < vStar.size(); ++j) {
                    std::cout << vStar(j) << " ";
                }
                std::cout <<"]\n";
            } else {
                linOpt.argmin(Phi, W, rep, d, vStar);
            }
            if (static_cast<V>(-1.) * this->fn->subgradient(xOld).dot(vStar - xOld) <= epsilon) {
                break;
            }
            if (!doLineSearch) {
                gamma = gamma0;
            } else {
                gamma = lineSearch.findOptimalPoint(xOld, vStar);
            }
            xNew = gamma * xOld + (1 - gamma) * vStar;
        }
        return xNew;
    }


    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(std::vector<Vector<V>> &Phi,
                                    Vector<V> &xIn,
                                    PolytopeType rep,
                                    bool doLineSearch) {
        std::vector<Vector<V>> W0;
        return this->argmin(Phi, W0, xIn, rep, doLineSearch);
    }


    template<typename V>
    FrankWolfe<V>::FrankWolfe(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
            : fn(f) {}

    template
    class FrankWolfe<double>;
}
