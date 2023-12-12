//
// Created by guoxin on 1/12/23.
//


#include "AchievabilityQuery.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void AchievabilityQuery<T, I>::query() {

        assert(this->data_.rowGroupIndices.size() == this->data_.colCount + 1);
        //mopmc::value_iteration::gpu::CudaValueIterationHandler<double> cudaVIHandler(this->data_);
        mopmc::optimization::optimizers::LinOpt<T> linOpt;
        PolytopeType rep = Closure;

        this->VIhandler->initialize();

        const uint64_t m = this->data_.objectiveCount; // m: number of objectives
        Vector<T> h = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());
        std::vector<Vector<T>> Phi, W;

        Vector<T> sgn(m); // optimisation direction
        for (uint_fast64_t i=0; i<sgn.size(); ++i) {
            sgn(i) = this->data_.isThresholdUpperBound[i] ? static_cast<T>(-1) : static_cast<T>(1);
        }

        Vector<T> r(m), w(m), w1(m + 1);
        std::vector<double> r_(m + 1), w_(m);

        const uint64_t maxIter{20};
        uint_fast64_t iter = 0;
        w.setConstant(static_cast<T>(1.0) / m); //initial direction
        bool achievable = true;
        T delta;

        while (iter < maxIter) {
            if (!Phi.empty()) {
                linOpt.findOptimalSeparatingDirection(Phi, rep, h, sgn, w1);
                w = VectorMap<T>(w1.data(), w1.size() - 1);
                delta = w1(w1.size() - 1);
                if (delta <= 0)
                    break;
            }

            for (uint_fast64_t i = 0; i < w.size(); ++i) {
                w_[i] = (double) (sgn(i) * w(i));
            }
            this->VIhandler->valueIteration(w_);
            r_ = this->VIhandler->getResults();
            r_.resize(m);
            for (uint_fast64_t i = 0; i < r_.size(); ++i) {
                r(i) = (T) r_[i];
            }
            Phi.push_back(r);
            W.push_back(w);

            Vector<T> wTemp = (sgn.array() * w.array()).matrix();
            if (wTemp.dot(h - r) > 0) {
                achievable = false;
                break;
            }
            //std::cout << "weighted value: " << cudaVIHandler.getResults()[m]<<"\n";
            ++iter;
        }
        this->VIhandler->exit();
        std::cout << "----------------------------------------------\n";
        std::cout << "Achievability Query terminates after " << iter << " iteration(s) \n";
        std::cout << "OUTPUT: " << std::boolalpha << achievable << "\n";
        std::cout << "----------------------------------------------\n";
    }

    template
    class AchievabilityQuery<double, int>;
}