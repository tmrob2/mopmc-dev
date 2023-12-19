#include "HybridQuery.h"
#include "mopmc-src/hybrid-computing/Problem.h"
#include "mopmc-src/hybrid-computing/Looper.h"
#include "mopmc-src/optimizers/LinOpt.h"


namespace mopmc::queries {

template<typename T, typename I>
void HybridQuery<T, I>::query() {
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
    Vector<T> sgn(m); // optimisation direction
    for (uint_fast64_t i=0; i<sgn.size(); ++i) {
        sgn(i) = this->data_.isThresholdUpperBound[i] ? static_cast<T>(-1) : static_cast<T>(1);
    }
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
    // next step is to group the 
    threadPool.scheduleProblems(w_, hybrid::ThreadSpecialisation::GPU);

    threadPool.stop();
    std::cout << "ThreadPool stopped\n";

    std::cout << "----------------------------------------------\n";
    std::cout << "@_@ Achievability Query terminates after " << iteration << " iteration(s) \n";
    std::cout << "*OUTPUT*: "<< std::boolalpha<< achievable<< "\n";
    std::cout << "----------------------------------------------\n";
}

template class HybridQuery<double, int>;

}

