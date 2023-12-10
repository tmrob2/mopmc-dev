//
// Created by guoxin on 21/11/23.
//

#include "TotalReLU.h"
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace mopmc::optimization::convex_functions {

    template<typename V>
    TotalReLU<V>::TotalReLU(const Vector<V> &c) : BaseConvexFunction<V>(c) {
        this->smooth = false;
    }

    template<typename V>
    V TotalReLU<V>::value1(const std::vector<V> &x) {

        std::vector<V> z = x;
        Vector<V> z1 = VectorMap<V>(z.data(), z.size());
        return value(z1);

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
    std::vector<V> TotalReLU<V>::subgradient1(const std::vector<V> &x) {

        std::vector<V> z = x;
        Vector<V> z1 = VectorMap<V>(z.data(), z.size());
        Vector<V> z2 = subgradient(z1);
        return std::vector<V>(z2.data(), z2.data() + z2.size());

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
    V TotalReLU<V>::value(const Vector<V> &x) {
        if (x.size() != this->params_.size()) {
            throw std::runtime_error("Convex function input does not match its dimension.");
        }
        Vector<V> y(this->params_.size());
        for (uint_fast64_t i = 0; i < this->params_.size(); ++i) {
            y(i) = std::max(static_cast<V>(0.), this->params_(i) - x(i));
        }
        return y.sum();
    }

    template<typename V>
    Vector<V> TotalReLU<V>::subgradient(const Vector<V> &x) {
        if (x.size() != this->params_.size()) {
            throw std::runtime_error("Convex function input does not match its dimension.");
        }
        Vector<V> dy(this->params_.size());
        for (uint_fast64_t i = 0; i < this->params_.size(); ++i) {
            if (x(i) < this->params_(i)) {
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