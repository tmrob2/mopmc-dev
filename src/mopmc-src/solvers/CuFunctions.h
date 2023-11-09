//
// Created by guoxin on 8/11/23.
//

#ifndef MOPMC_CUFUNCTIONS_H
#define MOPMC_CUFUNCTIONS_H

namespace mopmc::functions::cuda{

    int selectStateValuesLauncher(double *y, double *x, int *enabledActions, int* pi, int arrCount);

    int aggregateLauncher(const double *w, const double *x, double *z, int k, int n, int m);

};

#endif //MOPMC_CUFUNCTIONS_H
