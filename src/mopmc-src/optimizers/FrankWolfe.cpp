//
// Created by guoxin on 24/11/23.
//

#include <cmath>
#include <iostream>
#include "FrankWolfe.h"
#include "../convex-functions/BaseConvexFunction.h"

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
        Vector<V> xOld(m), xNew = xIn, vStar(m);

        V epsilon = 1.e-3;
        V gamma;
        V gamma0 = static_cast<V>(0.1);
        int maxIter = 5000;
        int i;
        for (i = 0; i < maxIter; ++i) {
            /*{
                std::cout << "**xNew in FrankWolfe**: [";
                for (int j = 0; j < xNew.size(); ++j) {
                   std::cout << xNew(0) <<" ";
                } std::cout << "]\n";
            }*/

            xOld = xNew;
            Vector<V> d = this->fn->subgradient(xOld);
            if (rep == Vertex) {
                linOpt.optimizeVtx(Phi, rep, d, vStar);
            }
            if (rep == Halfspace) {
                linOpt.optimizeHlsp(Phi, W, rep, d, vStar);
            }

            //V gap = cosine<V>(static_cast<V>(-1.) * this->fn->subgradient(xOld), vStar - xOld, static_cast<V>(0.));
            V gap = static_cast<V>(-1.) * this->fn->subgradient(xOld).dot(vStar - xOld);
            /*{
                std::cout << "**Frank-Wolfe** decent gap: "
                          << gap
                          << " at iteration " << i << " and epsilon is " << epsilon <<"\n";
                std::cout << "**Frank-Wolfe** vStar: [";
                for (int j = 0; j < vStar.size(); ++j) {
                    std::cout << vStar[j] << " ";
                } std::cout << "]\n";

            }*/
            if (gap <= epsilon ) {
                break;
            }
            if (!doLineSearch) {
                gamma = gamma0 / std::log(i+2); //static_cast<V>(i+1); //std::pow(static_cast<V>(1.01), i) ;
            } else {
                gamma = gamma0 / std::log(i+2); static_cast<V>(i+1); //std::pow(static_cast<V>(1.01), i) ;
                //gamma = lineSearch.findOptimalPoint(xOld, vStar);
            }
            xNew = (1-gamma) * xOld + gamma * vStar;
            //std::cout << "fn(xNew): " << this->fn->value(xNew) <<"\n";
        }
        std::cout << "**Frank-Wolfe** stops at iteration " << i <<"\n";
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

    template class FrankWolfe<double>;
}
