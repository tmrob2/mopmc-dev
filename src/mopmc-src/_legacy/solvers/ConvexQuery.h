//
// Created by thomas on 5/09/23.
//

#ifndef MOPMC_CONVEXQUERY_H
#define MOPMC_CONVEXQUERY_H

#include <vector>
#include <cstdint>
#include <unordered_map>
#include <Eigen/Dense>

namespace mopmc::solver::convex {

    // TODO probably best to turn convex query into a class and call member methods which contain
    //  all of the problem attributes rather than long signatures
    enum ARITH {
        ADD,
        SUB
    };

    template <typename T>
    struct hashFunction {
        size_t operator()(const std::vector<T> &myVector) const {
            std::hash<int> hasher;
            size_t answer = 0;

            for (T val : myVector) {
                answer ^= hasher(val) + 0x9e3779b9 +
                          (answer << 6) + (answer >> 2);
            }
            return answer;
        }
    };

    template<typename T>
    T l1Norm(std::vector<T> &x);

    template<typename T>
    T diff(std::vector<T> &fxStar, std::vector<T> &fzStar);

    template<typename T>
    std::vector<T>
    frankWolfe(std::vector<T> (*gradient)(Eigen::Matrix<T, Eigen::Dynamic, 1> &), std::vector<T> &initialPoint,
               uint64_t numberIterations, std::vector<std::vector<T>> &W,
               std::vector<std::vector<T>> &Phi, std::vector<T> &constraints);

    template<typename T>
    std::vector<T> getS_t(std::vector<T> (*gradient)(Eigen::Matrix<T, Eigen::Dynamic, 1> &), std::vector<T> &x,
                          std::vector<std::vector<T>> &W, std::vector<std::vector<T>> &Phi,
                          std::vector<T> &constraints);

    template<typename T>
    std::vector<T> ReLU(std::vector<T> &c, std::vector<T> &x){

        std::vector<T> output(c.size());
        for (uint_fast64_t i = 0 ; i < c.size() ; ++i ){
            output[i] = std::max(static_cast<T>(0.), c[i] - x[i]);
        }
        return output;
    }

    template<typename T>
    std::vector<T> reluGradient(Eigen::Matrix<T, Eigen::Dynamic, 1> &x) {
        std::vector<T> rtn(x.size());
        T zero_ = static_cast<T>(0.);
        T one_ = static_cast<T>(1.);
        for (uint_fast64_t i = 0; i < x.size(); ++i) {
            if (x[i] > zero_) {
                rtn[i] = one_;
            } else {
                rtn[i] = zero_;
            }
        }
        return rtn;
    }

    template <typename T>
    std::vector<T> computeNewW(std::vector<T> &x);

    template <typename T>
    std::vector<T> projectPointToNearestPlane(std::vector<T> &x,
                                              std::vector<std::vector<T>> &Phi,
                                              std::vector<std::vector<T>> &W,
                                              uint64_t l);

    template <typename T>
    std::vector<T> projectedGradientDescent(
            std::vector<T> (*gradient)(Eigen::Matrix<T, Eigen::Dynamic, 1>&),
            std::vector<T> &initPoint,
            T gamma,
            uint_fast64_t iterations,
            std::vector<std::vector<T>> &Phi,
            std::vector<std::vector<T>> &W,
            uint64_t l,
            std::vector<T> &targetPoint,
            T threshold);

}
#endif //MOPMC_CONVEXQUERY_H
