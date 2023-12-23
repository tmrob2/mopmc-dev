//
// Created by guoxin on 23/11/23.
//

#include "SignedKLEuclidean.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>

namespace mopmc::optimization::convex_functions {


    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(const std::vector<V> &c, const std::vector<bool> &isProb)
        : c_(c), asProbabilities(isProb) {
        if (c_.size() != asProbabilities.size()) {
            throw std::runtime_error("<to be inserted>");
        }
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(const std::vector<V> &c) : c_(c) {
        asProbabilities = std::vector<bool>(c_.size(), false);
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(const Vector<V> &e, const std::vector<bool> &isProb)
        : BaseConvexFunction<V>(e) {
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(const Vector<V> &e) : BaseConvexFunction<V>(e) {}

    template<typename V>
    V SignedKLEuclidean<V>::value(const Vector<V> &x) {

        if (this->parameters.size() != x.size()) {
            throw std::runtime_error("input dimension mismatch");
        }
        V out = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < this->asProbabilities.size(); ++i) {
            if (this->asProbabilities[i]) {
                out += sign_leq(x(i), this->parameters(i)) * klDivergence(x(i), this->parameters(i));
            } else {
                out += sign_leq(x(i), this->parameters(i)) * squaredDiff(x(i), this->parameters(i));
            }
        }
        return out;
    }

    template<typename V>
    Vector<V> SignedKLEuclidean<V>::subgradient(const Vector<V> &x) {
        if (this->parameters.size() != x.size()) {
            throw std::runtime_error("input dimension mismatch");
        }
        Vector<V> out(x.size());
        for (uint_fast64_t i = 0; i < this->asProbabilities.size(); ++i) {
            if (this->asProbabilities[i]) {
                out(i) = d_klDivergence(x(i), this->parameters(i));
            } else {
                out(i) = d_squaredDiff(x(i), this->parameters(i));
            }
        }
        return out;
    }

    template
    class SignedKLEuclidean<double>;
}
