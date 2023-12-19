#ifndef MOPMC_HYBRIDQUERY_H
#define MOPMC_HYBRIDQUERY_H

#include <Eigen/Dense>
#include "BaseQuery.h"
#include "../solvers/CudaValueIteration.cuh"
#include "mopmc-src/hybrid-computing/Problem.h"

namespace mopmc::queries {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;

    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename T, typename I>
    class HybridQuery : public BaseQuery<T, I>{
    public:

        explicit HybridQuery(const mopmc::Data<T,I> &data, 
                            hybrid::ThreadSpecialisation archPref = hybrid::ThreadSpecialisation::CPU) 
                            : BaseQuery<T, I>(data), threadSpec(archPref) {};
        void query() override;
    private:
        hybrid::ThreadSpecialisation threadSpec;
    };

}


#endif //MOPMC_HYBRIDQUERY_H