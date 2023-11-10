//
// Created by guoxin on 8/11/23.
//

#ifndef MOPMC_CUFUNCTIONS_H
#define MOPMC_CUFUNCTIONS_H

namespace mopmc::functions::cuda{

    int aggregateLauncher(const double *w, const double *x, double *z, int k, int n, int m);

    int maxValueLauncher(double *y, double *x, int *enabledActions, int* pi, int arrCount);

    int absLauncher(const double *x, int k);

};

#endif //MOPMC_CUFUNCTIONS_H
