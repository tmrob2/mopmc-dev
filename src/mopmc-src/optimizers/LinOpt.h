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
#include "lp_lib.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    class LinOpt {
    public:
        int argmin(std::vector<std::vector<V>> &Phi, std::vector<std::vector<V>> &W,
                              PolytopeRep &rep, std::vector<V> d, std::vector<V> &optValues);

        int argmin(std::vector<std::vector<V>> &Phi, PolytopeRep &rep,
                   std::vector<V> d, std::vector<V> &optValues);

    };
}

#endif //MOPMC_LINOPT_H
