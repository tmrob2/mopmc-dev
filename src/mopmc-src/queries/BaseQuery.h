//
// Created by guoxin on 27/11/23.
//

#ifndef MOPMC_BASEQUERY_H
#define MOPMC_BASEQUERY_H

#include <storm/api/storm.h>
#include "../Data.h"

namespace mopmc::queries {

    template<typename T>
    class BaseQuery {
    public:

        explicit BaseQuery() = default;
        explicit BaseQuery(const mopmc::Data<T,uint64_t> &data): data_(data){};

        virtual void query() = 0 ;

        mopmc::Data<T, uint64_t> data_;
        storm::Environment env_;
    };



}

#endif //MOPMC_BASEQUERY_H
