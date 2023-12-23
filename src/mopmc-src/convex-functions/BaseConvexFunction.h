//
// Created by guoxin on 21/11/23.
//

#ifndef MOPMC_BASECONVEXFUNCTION_H
#define MOPMC_BASECONVEXFUNCTION_H

#include <Eigen/Dense>
#include <utility>

namespace mopmc::optimization::convex_functions {

    template<typename V>
    using Vector = Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class BaseConvexFunction {
    public:
        explicit BaseConvexFunction() = default;
        explicit BaseConvexFunction(const Vector<V> &params) : parameters(params), dimension(params.size()) {}
        explicit BaseConvexFunction(const uint64_t dim) : dimension(dim){}

        virtual V value(const Vector<V> &x) = 0;
        virtual Vector<V> subgradient(const Vector<V> &x) = 0;
        Vector<V> gradient(const Vector<V> &x);

        V value1(const std::vector<V> &x);
        std::vector<V> subgradient1(const std::vector<V> &x);
        std::vector<V> gradient1(const std::vector<V> &x);

        uint64_t dimension{};
        Vector<V> parameters;
        bool smooth{};
    };
}// namespace mopmc::optimization::convex_functions

#endif//MOPMC_BASECONVEXFUNCTION_H
