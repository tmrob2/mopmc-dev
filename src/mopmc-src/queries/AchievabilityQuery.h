//
// Created by guoxin on 2/11/23.
//

#ifndef MOPMC_ACHIEVABILITYQUERY_H
#define MOPMC_ACHIEVABILITYQUERY_H

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <thread>
#include "BaseQuery.h"
#include "../Data.h"
#include "../solvers/CudaValueIteration.cuh"
#include "../optimizers/LinOpt.h"
#include "../optimizers/PolytopeTypeEnum.h"
#include "mopmc-src/hybrid-computing/Looper.h"
#include "mopmc-src/hybrid-computing/Problem.h"

namespace mopmc::queries {

    template<typename V>
    using Vector =  Eigen::Matrix<V, Eigen::Dynamic, 1>;
    template<typename V>
    using VectorMap = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>;

    template<typename T, typename I>
    class AchievabilityQuery : public BaseQuery<T, I>{
    public:
        explicit AchievabilityQuery(const mopmc::Data<T,I> &data) : BaseQuery<T, I>(data) {};
        void query() override;
        void hybridQuery(hybrid::ThreadSpecialisation archPref);
    };

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
        assert(this->data_.rowGroupIndices.size()==k+1);
        Vector<T> h = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());
        std::vector<std::vector<T>> rho(m);
        std::vector<T> rho_flat(n * m);
        std::vector<Vector<T>> Phi;
        std::vector<Vector<T>> W;
        Vector<T> sgn(m);
        sgn.fill(static_cast<T>(-1));
        std::vector<T> w1;
        Vector<T> dirVec(m + 1);
        bool achievable = true;
        Vector<T> r(m);
        std::vector<double> r_(m+1);
        Vector<T> w(m);
        //----(initial w for testing)----
        //w << 0.5, 0.5;
        w.setConstant(static_cast<T>(1.0) / m);
        //-------------------------------
        std::vector<double> w_(m);
        Vector<T> sw (m);
        T delta;
        PolytopeType rep = Closure;
        const uint64_t maxIter{20};

        uint_fast64_t iteration = 0;
        while (iteration < maxIter) {
             if (!Phi.empty()) {
                 linOpt.optimize(Phi, rep, h, sgn, dirVec);
                 assert(dirVec.size() == m + 1);
                 w = VectorMap<T> (dirVec.data(), dirVec.size() - 1);
            }

            delta = dirVec(dirVec.size() - 1);
            if (delta <= 0)
                break;
            for (uint_fast64_t i=0; i < w.size(); ++i) {
                w_[i] = (double) (sgn(i) * w (i));
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
        std::cout << "*OUTPUT*: "<< std::boolalpha<< achievable<< "\n";
        std::cout << "----------------------------------------------\n";
    }

    template<typename T, typename I>
    void AchievabilityQuery<T, I>::hybridQuery(hybrid::ThreadSpecialisation archPref) {
        // =============================
        // Multithreading approach
        typedef hybrid::ThreadProblem<T> Pr;
        std::vector<std::unique_ptr<hybrid::CLooper<T>>> threads(2); 
        threads[0] = std::move(std::make_unique<hybrid::CLooper<T>>(0, hybrid::ThreadSpecialisation::CPU, this->data_));
        threads[1] = std::move(std::make_unique<hybrid::CLooper<T>>(1, hybrid::ThreadSpecialisation::GPU, this->data_));

        auto threadPool = hybrid::CLooperPool<T>(std::move(threads));

        std::cout << "Starting the thread pool: \n";
        threadPool.run();

        while(!threadPool.running()) {
            // do nothing
        }

        std::cout << "Thread pool started!\n";
        // initialise the problem queue
        std::vector<hybrid::ThreadProblem<T>> problems;
        // =============================
        // End of multithreading setup


        // =============================
        // Initialise the problem

        mopmc::optimization::optimizers::LinOpt<T> linOpt;

        //variable definitions
        const uint64_t m = this->data_.objectiveCount; // m: number of objectives
        const uint64_t n = this->data_.rowCount; // n: number of choices / state-action pairs
        const uint64_t k = this->data_.colCount; // k: number of states
        assert(this->data_.rowGroupIndices.size()==k+1);
        Vector<T> h = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());
        std::vector<std::vector<T>> rho(m);
        std::vector<T> rho_flat(n * m);
        std::vector<Vector<T>> Phi;
        std::vector<Vector<T>> W;
        Vector<T> sgn(m);
        sgn.fill(static_cast<T>(-1));
        std::vector<T> w1;
        Vector<T> dirVec(m + 1);
        bool achievable = true;
        Vector<T> r(m);
        std::vector<double> r_(m+1);
        Vector<T> w(m);
        //----(initial w for testing)----
        //w << 0.5, 0.5;
        w.setConstant(static_cast<T>(1.0) / m);
        //-------------------------------
        std::vector<double> w_(m);
        Vector<T> sw (m);
        T delta;
        PolytopeType rep = Closure;
        const uint64_t maxIter{20};

        uint_fast64_t iteration = 0;
        for (uint_fast64_t i=0; i < w.size(); ++i) {
            w_[i] = (double) (sgn(i) * w (i));
        }

        // =============================
        // begin solving the problem

        // To construct a scheduler we can either use the cpu or gpu via the hybrid framework
        // A scheduler 'problem' is just a single problem pushed to the threads and solved on
        // either the GPU or CPU
        hybrid::ThreadProblem<T> schPr(0, w_, hybrid::ThreadSpecialisation::GPU, hybrid::Problem::Scheduler);
        problems.push_back(schPr);
        threadPool.solve(problems);

        //cudaVIHandler.valueIteration(w_);
        //r_ = cudaVIHandler.getResults();
        auto solutions = threadPool.getSolutions();
        
        threadPool.stop();
        std::cout << "ThreadPool stopped\n";
            
        std::cout << "----------------------------------------------\n";
        std::cout << "@_@ Achievability Query terminates after " << iteration << " iteration(s) \n";
        std::cout << "*OUTPUT*: "<< std::boolalpha<< achievable<< "\n";
        std::cout << "----------------------------------------------\n";
    }

    //template class AchievabilityQuery<double, uint64_t>;
    template class AchievabilityQuery<double, int>;
}


#endif //MOPMC_ACHIEVABILITYQUERY_H
