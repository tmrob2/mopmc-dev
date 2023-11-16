//
// Created by guoxin on 8/11/23.
//

#ifndef MOPMC_CUFUNCTIONS_H
#define MOPMC_CUFUNCTIONS_H

namespace mopmc::functions::cuda{

    int aggregateLauncher(const double *w, const double *x, double *y, int numRows, int numObjs);

    int maxValueLauncher1(double *y, double *x, int *enabledActions, int* pi, int arrCount, int numRows);

    int maxValueLauncher2(double *y, double *x, int *enabledActions, int* pi, int* bpi, int arrCount);

    int maskingLauncher(const int* csrOffsets, const int *rowGroupIndices, const int *row2RowGroupIndices,
                        const int* pi, int* maskVec, int arrCount);
    int row2RowGroupLauncher(const int *row2RowGroupMapping, int *x, int arrCount);

    //Predicate functor
    template<typename T>
    struct is_not_zero
    {
        __host__ __device__
        bool operator()(const T x)
        {
            return (x != 0);
        }
    };

    int absLauncher(const double *x, int k);

};

#endif //MOPMC_CUFUNCTIONS_H
