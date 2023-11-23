//
// Created by guoxin on 21/11/23.
//

#include "TotalReLU.h"
#include <numeric>
#include <stdexcept>

namespace mopmc::optimisation::convex_functions {


    template<typename V>
    TotalReLU<V>::TotalReLU(std::vector<V> &c) : c_(c) {}

    template<typename V>
    V TotalReLU<V>::value(std::vector<V> &x) {

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
    std::vector<V> TotalReLU<V>::subgradient(std::vector<V> &x) {

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

    template
    class TotalReLU<double>;
}