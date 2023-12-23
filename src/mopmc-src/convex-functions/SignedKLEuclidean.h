//
// Created by guoxin on 23/11/23.
//

#ifndef MOPMC_SIGNEDKLEUCLIDEAN_H
#define MOPMC_SIGNEDKLEUCLIDEAN_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include "BaseConvexFunction.h"

namespace mopmc::optimization::convex_functions {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class SignedKLEuclidean : public BaseConvexFunction<V> {
    public:
        explicit SignedKLEuclidean(const std::vector<V> &c);
        SignedKLEuclidean(const std::vector<V> &c, const std::vector<bool> &isProb);

        explicit SignedKLEuclidean(const Vector<V> &e);
        SignedKLEuclidean(const Vector<V> &e, const std::vector<bool> &isProb);

        V value(const Vector<V> &x) override;

        Vector<V> subgradient(const Vector<V> &x) override;

        std::vector<V> c_;
        std::vector<bool> asProbabilities;
    };

    template<typename V>
    V klDivergence(V x, V c) {
        if (x < static_cast<V>(0.) || x > static_cast<V>(1.) || c <= static_cast<V>(0.) || c >= static_cast<V>(1.)) {
            throw std::runtime_error("<to be inserted>");
        }
        return x * (std::log(x / c)) + (1 - x) * (std::log((1 - x) / (1 - c)));
    }

    template<typename V>
    V d_klDivergence(V x, V c) {
        if (x <= static_cast<V>(0.) || x >= static_cast<V>(1.) || c <= static_cast<V>(0.) || c >= static_cast<V>(1.)) {
            throw std::runtime_error("<to be inserted>");
        }
        return std::log(x / c) - std::log((1 - x) / (1 - c));
    }

    template<typename V>
    V squaredDiff(V x, V c) {
        return std::pow(x - c, 2);
    }

    template<typename V>
    V d_squaredDiff(V x, V c) {
        return 2 * (x - c);
    }

    template<typename V>
    V sign_leq(V x, V y) {
        if (x <= y) {
            return static_cast<V>(1.);
        } else {
            return static_cast<V>(0.);
        }
    }

    template<typename V>
    V klNorm(std::vector<V> x, std::vector<V> y) {
        assert (x.size() == y.size());
        V out = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            out += klDivergence(x[i], y[i]);
        }
        return out;
    }

    template<typename V>
    V squaredL2Norm(std::vector<V> x, std::vector<V> y) {
        assert (x.size() == y.size());
        V out = static_cast<V>(0.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            out += std::pow(x[i] - y[i], 2);
        }
        return out;
    }

};


#endif //MOPMC_SIGNEDKLEUCLIDEAN_H
