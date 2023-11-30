//
// Created by guoxin on 27/11/23.
//

#ifndef MOPMC_BASEQUERY_H
#define MOPMC_BASEQUERY_H

#include <storm/api/storm.h>
#include "../Data.h"

namespace mopmc::queries {

    template<typename T, typename I>
    class BaseQuery {
    public:

        explicit BaseQuery() = default;
        explicit BaseQuery(const mopmc::Data<T,I> &data): data_(data){};

        virtual void query() = 0 ;

        mopmc::Data<T, I> data_;
        storm::Environment env_;
    };



}

#endif //MOPMC_BASEQUERY_H
