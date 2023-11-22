//
// Created by guoxin on 21/11/23.
//

#ifndef MOPMC_BASECONVEXFUNCTION_H
#define MOPMC_BASECONVEXFUNCTION_H

namespace mopmc::optimisation::convex_functions {

    template<typename V>
    class BaseConvexFunction {
    public:

        virtual V value(std::vector<V>) = 0;
        virtual std::vector<V> subgradient(std::vector<V>) = 0;
    };
}

#endif //MOPMC_BASECONVEXFUNCTION_H
