//
// Created by guoxin on 20/11/23.
//

#include "Data.h"

namespace mopmc {


    template<typename V, typename I>
    void mopmc::Data<V, I>::bar(const V &t) {}

    template
    struct Data<double, uint64_t>;

}
