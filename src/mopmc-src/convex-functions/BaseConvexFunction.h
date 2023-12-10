//
// Created by guoxin on 21/11/23.
//

#ifndef MOPMC_BASECONVEXFUNCTION_H
#define MOPMC_BASECONVEXFUNCTION_H

#include <Eigen/Dense>
#include <utility>

namespace mopmc::optimization::convex_functions {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class BaseConvexFunction {
    public:

        explicit BaseConvexFunction()= default;
        explicit BaseConvexFunction(const Vector<V> &params) : params_(params){}
        explicit BaseConvexFunction(const Vector<V> &params, const std::vector<bool> &probs)
            : params_(params), probs_(probs){}
        explicit BaseConvexFunction(const VectorMap<V> &params) : params_(params){}
        explicit BaseConvexFunction(const VectorMap<V> &params, const std::vector<bool> &probs)
            : params_(params), probs_(probs){}

        virtual V value(const Vector<V> &x) = 0;
        virtual V value1(const std::vector<V> &x) = 0;

        virtual Vector<V> subgradient(const Vector<V> &x) = 0;
        virtual std::vector<V> subgradient1(const std::vector<V> &x) = 0;

        //virtual Vector<V> gradient(const Vector<V> &x) = 0;

        Vector<V> params_;
        std::vector<bool> probs_;
        bool smooth{};

    };
}

#endif //MOPMC_BASECONVEXFUNCTION_H
