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
        : c_(c), isProb2_(isProb) {
        if (c_.size() != isProb2_.size()) {
            throw std::runtime_error("<to be inserted>");
        }
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(const std::vector<V> &c) : c_(c) {
        isProb2_ = std::vector<bool>(c_.size(), false);
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(const Vector<V> &e, const std::vector<bool> &isProb)
        : BaseConvexFunction<V>(e, isProb) {
        if (this->params_.size() != this->probs_.size()) {
            throw std::runtime_error("<to be inserted>");
        }
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(const Vector<V> &e) : BaseConvexFunction<V>(e) {}

    template<typename V>
    V SignedKLEuclidean<V>::value(const Vector<V> &x) {

        if (this->params_.size() != x.size()) {
            throw std::runtime_error("input dimension mismatch");
        }
        V out = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < this->probs_.size(); ++i) {
            if (this->probs_[i]) {
                out += sign_leq(x(i), this->params_(i)) * klDivergence(x(i), this->params_(i));
            } else {
                out += sign_leq(x(i), this->params_(i)) * squaredDiff(x(i), this->params_(i));
            }
        }
        return out;
    }

    template<typename V>
    Vector<V> SignedKLEuclidean<V>::subgradient(const Vector<V> &x) {
        if (this->params_.size() != x.size()) {
            throw std::runtime_error("input dimension mismatch");
        }
        Vector<V> out(x.size());
        for (uint_fast64_t i = 0; i < this->probs_.size(); ++i) {
            if (this->probs_[i]) {
                out(i) = d_klDivergence(x(i), this->params_(i));
            } else {
                out(i) = d_squaredDiff(x(i), this->params_(i));
            }
        }
        return out;
    }

    template
    class SignedKLEuclidean<double>;
}
