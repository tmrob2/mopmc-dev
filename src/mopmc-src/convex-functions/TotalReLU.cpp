//
// Created by guoxin on 21/11/23.
//

#include "TotalReLU.h"
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace mopmc::optimization::convex_functions {

    template<typename V>
    TotalReLU<V>::TotalReLU(std::vector<V> &c) : c_(c), BaseConvexFunction<V>() {}
    template<typename V>
    TotalReLU<V>::TotalReLU(Vector<V> &e) : BaseConvexFunction<V>(e) {}
    template<typename V>
    TotalReLU<V>::TotalReLU(VectorMap<V> &e) : BaseConvexFunction<V>(e) {}

    template<typename V>
    V TotalReLU<V>::value1(std::vector<V> &x) {

        if (x.size() != c_.size()) {
            throw std::runtime_error("Convex function input does not match its dimension.");
        }

        std::vector<V> y(c_.size());
        for (uint_fast64_t i = 0; i < c_.size(); ++i) {
            y[i] = std::max(static_cast<V>(0.), c_[i] - x[i]);
        }
        return std::reduce(y.begin(), y.end());
    }

    template<typename V>
    std::vector<V> TotalReLU<V>::subgradient1(std::vector<V> &x) {

        if (x.size() != c_.size()) {
            throw std::runtime_error("Convex function input does not match its dimension.");
        }

        std::vector<V> dy(c_.size());
        for (uint_fast64_t i = 0; i < c_.size(); ++i) {
            if (x[i] < c_[i]) {
                dy[i] = static_cast<V>(-1.);
            } else {
                dy[i] = static_cast<V>(0.);
            };
        }
        return dy;
    }

    template<typename V>
    V TotalReLU<V>::value(Vector<V> &x) {
        //std::cout << "x.size(): " << x.size() << ", e_.size(): " << this->e_.size() <<std::endl;
        if (x.size() != this->e_.size()) {
            throw std::runtime_error("Convex function input does not match its dimension.");
        }
        Vector<V> y(this->e_.size());
        for (uint_fast64_t i = 0; i < this->e_.size(); ++i) {
            y(i) = std::max(static_cast<V>(0.), this->e_(i) - x(i));
        }
        return y.sum();
    }

    template<typename V>
    Vector<V> TotalReLU<V>::subgradient(Vector<V> &x) {
        if (x.size() != this->e_.size()) {
            throw std::runtime_error("Convex function input does not match its dimension.");
        }
        Vector<V> dy(this->e_.size());
        for (uint_fast64_t i = 0; i < this->e_.size(); ++i) {
            if (x(i) < this->e_(i)) {
                dy(i) = static_cast<V>(-1.);
            } else {
                dy(i) = static_cast<V>(0.);
            };
        }
        return dy;
    }

    template
    class TotalReLU<double>;
}