//
// Created by guoxin on 24/11/23.
//

#include <cmath>
#include "FrankWolfe.h"
#include "../convex-functions/BaseConvexFunction.h"

namespace mopmc::optimization::optimizers{

    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(std::vector<Vector<V>> &Phi,
                                    std::vector<Vector<V>> &W,
                                    Vector<V> &xIn,
                                    PolytopeRep rep,
                                    bool doLineSearch) {
        if (rep==HRep) {
            assert (!W.empty());
        }
        mopmc::optimization::optimizers::LineSearch<V> lineSearch(this->fn);
        auto m = xIn.size();
        Vector<V> xOld(m), xNew = xIn;
        Vector<V> vStar(m);
        for (int i = 0; i < maxIter; ++i) {
            xOld = xNew;
            Vector<V> d = this->fn->subgradient(xOld);
            mopmc::optimization::optimizers::LinOpt<V> linOpt;
            if (static_cast<V>(-1.) * this->fn->subgradient(xOld).dot(vStar - xOld) <= epsilon) {
                break;
            }
            if (rep == VRep ) {
                linOpt.argmin(Phi, rep, d, vStar);
            } else {
                linOpt.argmin(Phi, W, rep, d, vStar);
            }
            if (!doLineSearch) {
                gamma = gamma0;
            } else {
                gamma = lineSearch.findOptimalPoint(xOld, vStar);
            }
            xNew = gamma * xOld + (1-gamma) * vStar;
        }
        return xNew;
    }


    template<typename V>
    Vector<V> FrankWolfe<V>::argmin(std::vector<Vector<V>> &Phi,
                                    Vector<V> &xIn,
                                    PolytopeRep rep,
                                    bool doLineSearch) {
        std::vector<Vector<V>> W0;
        return this->argmin(Phi, W0, xIn, rep, doLineSearch);
    }


    template<typename V>
    FrankWolfe<V>::FrankWolfe(mopmc::optimization::convex_functions::BaseConvexFunction<V> *f)
            : fn(f){}

    template class FrankWolfe<double>;
}
