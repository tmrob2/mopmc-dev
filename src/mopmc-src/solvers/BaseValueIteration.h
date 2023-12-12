//
// Created by guoxin on 12/12/23.
//

#ifndef MOPMC_BASEVALUEITERATION_H
#define MOPMC_BASEVALUEITERATION_H

#include <vector>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>

namespace mopmc::value_iteration {

    template<typename V>
    class BaseVIHandler {
    public:

        virtual int initialize() {return  0;};
        virtual int exit() {return 0;}

        virtual int valueIteration(const std::vector<double> &w) {return 0;}

        virtual const std::vector<double> &getResults() const = 0;

    };
}

#endif //MOPMC_BASEVALUEITERATION_H
