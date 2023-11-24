//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_FRANKWOLFE_H
#define MOPMC_FRANKWOLFE_H

#include <vector>
#include <cassert>
#include <algorithm>
#include "PolytopeRepresentation.h"
#include "../convex-functions/BaseConvexFunction.h"
#include "LinOpt.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    class FrankWolfe {
    public:
        FrankWolfe(V epsilon, V gamma, int maxIter);

        std::vector<V> argmin(mopmc::optimization::convex_functions::BaseConvexFunction<V> &f,
                 std::vector<std::vector<V>> &Phi,
                 std::vector<std::vector<V>> &W,
                 std::vector<V> &xin,
                 PolytopeRep rep,
                 bool lineSearch);

        std::vector<V> argmin(mopmc::optimization::convex_functions::BaseConvexFunction<V> &f,
                 std::vector<std::vector<V>> &Phi,
                 std::vector<V> &xin,
                 PolytopeRep rep,
                 bool lineSearch);

        //std::vector<V> minValues{};
        V epsilon_;
        V gamma_ = static_cast<V>(2.0);
        int maxIter_ = 100;
    };

    template<typename V>
    std::vector<V> FrankWolfe<V>::argmin(convex_functions::BaseConvexFunction<V> &f, std::vector<std::vector<V>> &Phi,
                            std::vector<std::vector<V>> &W, std::vector<V> &xin, PolytopeRep rep, bool lineSearch) {
        if (rep==HRep) {
            assert (!W.empty());
        }
        std::vector<V> xOld, xNew = xin;
        std::vector<V> vStar(xin.size());
        V gamma1;
        for (int i = 0; i < maxIter_; ++i) {
            xOld = xNew;
            std::vector<V> d = f.subgradient(xOld);
            mopmc::optimization::optimizers::LinOpt<V> linOpt;
            // todo
            //if (static_cast(-1.)* f.subgradient(xOld) ( ))
            if (rep == VRep ) {
                linOpt.argmin(Phi, rep, d, vStar);
            } else {
                linOpt.argmin(Phi, W, rep, d, vStar);
            }
            if (!lineSearch) {
                gamma1 = gamma_;
            } else {
                //todo
                gamma1 = gamma_;
            }
            //std::bind1st(std::multiplies<T>(), 3))
            std::transform(xOld.begin(), xOld.end(), xOld.begin(), std::bind(std::multiplies<V>(),
                    std::placeholders::_1, gamma_));
            std::transform(vStar.begin(), vStar.end(), vStar.begin(), std::bind(std::multiplies<V>(),
                    std::placeholders::_1, static_cast<V>(1.)-gamma_));
            std::transform(xOld.begin(), xOld.end(), vStar.begin(), xNew.begin(), std::plus<V>());
        }
        return xNew;
    }

    template<typename V>
    std::vector<V> FrankWolfe<V>::argmin(convex_functions::BaseConvexFunction<V> &f, std::vector<std::vector<V>> &Phi,
                            std::vector<V> &xin, PolytopeRep rep, bool lineSearch) {
        std::vector<std::vector<V>> W = std::vector<std::vector<V>>();
        return this->argmin(f, Phi, W, xin, rep, lineSearch);
    }

    template<typename V>
    FrankWolfe<V>::FrankWolfe(V epsilon, V gamma, int maxIter): epsilon_(epsilon), gamma_(gamma), maxIter_(maxIter) {}

}


#endif //MOPMC_FRANKWOLFE_H
