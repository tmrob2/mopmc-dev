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