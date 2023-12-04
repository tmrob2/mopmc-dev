//
// Created by guoxin on 1/12/23.
//


#include "AchievabilityQuery.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void AchievabilityQuery<T, I>::query() {

        mopmc::value_iteration::gpu::CudaValueIterationHandler<double> cudaVIHandler(
                this->data_.transitionMatrix,
                this->data_.rowGroupIndices,
                this->data_.row2RowGroupMapping,
                this->data_.flattenRewardVector,
                this->data_.defaultScheduler,
                this->data_.initialRow,
                this->data_.objectiveCount
        );
        cudaVIHandler.initialise();

        mopmc::optimization::optimizers::LinOpt<T> linOpt;

        //variable definitions
        const uint64_t m = this->data_.objectiveCount; // m: number of objectives
        const uint64_t n = this->data_.rowCount; // n: number of choices / state-action pairs
        const uint64_t k = this->data_.colCount; // k: number of states
        assert(this->data_.rowGroupIndices.size() == k + 1);
        Vector<T> h = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());
        std::vector<std::vector<T>> rho(m);
        std::vector<T> rho_flat(n * m);
        std::vector<Vector<T>> Phi;
        std::vector<Vector<T>> W;
        Vector<T> sgn(m);
        for (uint_fast64_t i=0; i<sgn.size(); ++i) {
            sgn(i) = this->data_.optDirections[i] ? static_cast<T>(-1) : static_cast<T>(1);
        }
        std::vector<T> w1;
        Vector<T> dirVec(m + 1);
        bool achievable = true;
        Vector<T> r(m);
        std::vector<double> r_(m + 1);
        Vector<T> w(m);
        //----(initial w for testing)----
        //w << 0.5, 0.5;
        w.setConstant(static_cast<T>(1.0) / m);
        //-------------------------------
        std::vector<double> w_(m);
        Vector<T> sw(m);
        T delta;
        PolytopeType rep = Closure;
        const uint64_t maxIter{20};

        uint_fast64_t iteration = 0;
        while (iteration < maxIter) {
            if (!Phi.empty()) {
                linOpt.optimize(Phi, rep, h, sgn, dirVec);
                assert(dirVec.size() == m + 1);
                w = VectorMap<T>(dirVec.data(), dirVec.size() - 1);
                delta = dirVec(dirVec.size() - 1);
                if (delta <= 0)
                    break;
            }

            for (uint_fast64_t i = 0; i < w.size(); ++i) {
                w_[i] = (double) (sgn(i) * w(i));
            }
            cudaVIHandler.valueIteration(w_);
            r_ = cudaVIHandler.getResults();
            assert(dirVec.size() == m + 1);
            r_.resize(m);
            for (uint_fast64_t i = 0; i < r_.size(); ++i) {
                r(i) = (T) r_[i];
            }
            Phi.push_back(r);
            W.push_back(w);

            std::cout << "weighted value: " << cudaVIHandler.getResults()[m]<<"\n";

            sw = (sgn.array() * w.array()).matrix();
            if (sw.dot(h - r) > 0) {
                achievable = false;
                break;
            }
            ++iteration;
        }
        cudaVIHandler.exit();
        std::cout << "----------------------------------------------\n";
        std::cout << "@_@ Achievability Query terminates after " << iteration << " iteration(s) \n";
        std::cout << "*OUTPUT*: " << std::boolalpha << achievable << "\n";
        std::cout << "----------------------------------------------\n";
    }

    //template class AchievabilityQuery<double, uint64_t>;
    template
    class AchievabilityQuery<double, int>;
}