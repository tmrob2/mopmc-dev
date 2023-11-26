//
// Created by guoxin on 24/11/23.
//

#ifndef MOPMC_LINOPT_H
#define MOPMC_LINOPT_H

#include "PolytopeRepresentation.h"
#include <vector>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <Eigen/Dense>
#include "lp_lib.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename V>
    class LinOpt {
    public:

        int argmin(std::vector<Vector<V>> &Phi, std::vector<Vector<V>> &W, PolytopeRep &rep, Vector<V> d, Vector<V> &optValues);

        int argmin(std::vector<Vector<V>> &Phi, PolytopeRep &rep, Vector<V> d, Vector<V> &optValues);

    };
}

#endif //MOPMC_LINOPT_H
