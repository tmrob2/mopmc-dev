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
        explicit BaseConvexFunction(Vector<V> &e) : e_(e){}
        explicit BaseConvexFunction(Vector<V> &e, std::vector<bool> isProb) : e_(e), isProb_(std::move(isProb)){}
        explicit BaseConvexFunction(VectorMap<V> &e) : e_(e){}
        explicit BaseConvexFunction(VectorMap<V> &e, std::vector<bool> isProb) : e_(e), isProb_(std::move(isProb)){}

        virtual V value(std::vector<V> &x) = 0;
        virtual V value1(Vector<V> &x) = 0;

        virtual std::vector<V> subgradient(std::vector<V> &x) = 0;
        virtual Vector<V> subgradient1(Vector<V> &x) = 0;

        Vector<V> e_;
        std::vector<bool> isProb_;

    };
}

#endif //MOPMC_BASECONVEXFUNCTION_H
