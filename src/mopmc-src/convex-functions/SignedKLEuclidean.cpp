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
    SignedKLEuclidean<V>::SignedKLEuclidean(std::vector<V> &c, std::vector<bool> &isProb) : c_(c), isProb2_(isProb) {
        if (c_.size() != isProb2_.size()) {
            throw std::runtime_error("<to be inserted>");
        }
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(std::vector<V> &c) : c_(c) {
        isProb2_ = std::vector<bool>(c_.size(), false);
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(Vector<V> &e, std::vector<bool> &isProb) : BaseConvexFunction<V>(e, isProb) {
        if (this->e_.size() != this->isProb_.size()) {
            throw std::runtime_error("<to be inserted>");
        }
    }

    template<typename V>
    SignedKLEuclidean<V>::SignedKLEuclidean(Vector<V> &e) : BaseConvexFunction<V>(e) {}


    template<typename V>
    V SignedKLEuclidean<V>::value(std::vector<V> &x) {

        if (c_.size() != x.size()) {
            throw std::runtime_error("<to be inserted>");
        }
        V out = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < isProb2_.size(); ++i) {
            if (isProb2_[i]) {
                out += sign_leq(x[i], c_[i]) * klDivergence(x[i], c_[i]);
            } else {
                out += sign_leq(x[i], c_[i]) * squaredDiff(x[i], c_[i]);
            }
        }
        return out;
    }

    template<typename V>
    V SignedKLEuclidean<V>::value1(Vector<V> &x) {

        if (this->e_.size() != x.size()) {
            throw std::runtime_error("<to be inserted>");
        }
        V out = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < this->isProb_.size(); ++i) {
            if (this->isProb_[i]) {
                out += sign_leq(x(i), this->e_(i)) * klDivergence(x(i), this->e_(i));
            } else {
                out += sign_leq(x(i), this->e_(i)) * squaredDiff(x(i), this->e_(i));
            }
        }
        return out;
    }

    template<typename V>
    std::vector<V> SignedKLEuclidean<V>::subgradient(std::vector<V> &x) {
        if (c_.size() != x.size()) {
            throw std::runtime_error("<to be inserted>");
        }
        std::vector<V> out(x.size());
        for (uint_fast64_t i = 0; i < isProb2_.size(); ++i) {
            if (isProb2_[i]) {
                out[i] = d_klDivergence(x[i], c_[i]);
            } else {
                out[i] = d_squaredDiff(x[i], c_[i]);
            }
        }
        return out;
    }


    template<typename V>
    Vector<V> SignedKLEuclidean<V>::subgradient1(Vector<V> &x) {
        if (this->e_.size() != x.size()) {
            throw std::runtime_error("<to be inserted>");
        }
        Vector<V> out(x.size());
        for (uint_fast64_t i = 0; i < this->isProb_.size(); ++i) {
            if (this->isProb_[i]) {
                out(i) = d_klDivergence(x(i), this->e_(i));
            } else {
                out(i) = d_squaredDiff(x(i), this->e_(i));
            }
        }
        return out;
    }

    template
    class SignedKLEuclidean<double>;
}
