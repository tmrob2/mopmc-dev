//
// Created by thomas on 27/09/23.
//

#ifndef MOPMC_ACTIONSELECTION_H
#define MOPMC_ACTIONSELECTION_H

namespace mopmc::kernels{
int maxValueLauncher(double *y, double *x, int *enabledActions, int* pi, int arrCount);

double findMaxEps(double* y, int size, double maxDiff);

int launchPrintKernel(double printVal);
}


#endif //MOPMC_ACTIONSELECTION_H
