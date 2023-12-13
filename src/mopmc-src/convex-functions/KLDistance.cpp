//
// Created by guoxin on 13/12/23.
//

#include "KLDistance.h"
#include <iostream>

namespace mopmc::optimization::convex_functions {

    template<typename V>
    KLDistance<V>::KLDistance(const Vector<V> &c): BaseConvexFunction<V>(c) {
        this->smooth = true;
        /*
        for (uint_fast64_t i = 0; i < this->probs_.size(); ++i) {
            if (this->probs_[i]) {
                throw std::runtime_error("KL distance applies to probabilities only.");
            }
        }*/
    }

    template<typename V>
    V KLDistance<V>::value(const Vector<V> &x) {
        if (this->params_.size() != x.size()) {
            throw std::runtime_error("input dimension mismatch");
        }
        V y = static_cast<V>(0.);
        //std::cout << "x: " << x <<"\n";
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            y += klDivergence(x(i), this->params_(i));
            //std::cout << "klDivergence(): " << klDivergence(x(i), this->params_(i)) <<"\n";
        }
        return y;
    }

    template<typename V>
    Vector<V> KLDistance<V>::subgradient(const Vector<V> &x) {
        if (this->params_.size() != x.size()) {
            throw std::runtime_error("input dimension mismatch");
        }
        Vector<V> out(x.size());
        for (uint_fast64_t i = 0; i < this->probs_.size(); ++i) {
            out(i) = d_klDivergence(x(i), this->params_(i));
        }
        return out;
    }


    template<typename V>
    V KLDistance<V>::klDivergence(V x, V c) {
        if (x < 0 || x > 1 || c <= 0 || c >= 1) {
            throw std::runtime_error("probabilities' ranges must be >0 and <1.");
        }
        return x * (std::log(x / c)) + (static_cast<V>(1.) - x) * (std::log((static_cast<V>(1.) - x) / (static_cast<V>(1.) - c)));
    }

    template<typename V>
    V KLDistance<V>::d_klDivergence(V x, V c) {
        if (x <= static_cast<V>(0.) || x >= static_cast<V>(1.) || c <= static_cast<V>(0.) || c >= static_cast<V>(1.)) {
            throw std::runtime_error("probabilities' ranges must be >0 and <1.");
        }
        return std::log(x / c) - std::log((static_cast<V>(1.) - x) / (static_cast<V>(1.) - c));
    }

    template class KLDistance<double>;
}
