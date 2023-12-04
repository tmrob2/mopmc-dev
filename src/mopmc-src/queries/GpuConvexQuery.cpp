//
// Created by guoxin on 3/11/23.
//


#include <iostream>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "GpuConvexQuery.h"
#include "../solvers/ConvexQuery.h"
#include "../solvers/CudaValueIteration.cuh"
#include "../Data.h"
#include "../convex-functions/TotalReLU.h"
#include "../optimizers/FrankWolfe.h"
#include "../optimizers/PolytopeTypeEnum.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void GpuConvexQuery<T, I>::query() {

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

        //variable definitions
        const uint64_t m = this->data_.objectiveCount; // m: number of objectives
        const uint64_t n = this->data_.rowCount; // n: number of choices / state-action pairs
        const uint64_t k = this->data_.colCount; // k: number of states
        assert(this->data_.rowGroupIndices.size()==k+1);
        std::vector<T> h = this->data_.thresholds;
        Vector<T> h_ = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());

        //std::vector<std::vector<T>> rho(m);
        //std::vector<T> rho_flat(n*m);//rho: all reward vectors
        //Initialisation
        std::vector<std::vector<T>> Phi;
        std::vector<Vector<T>> Phi_;
        std::vector<std::vector<T>> W;
        std::vector<Vector<T>> W_;
        std::set<std::vector<T>> wSet;
        std::set<Vector<T>> wSet_;
        //vt, vb
        std::vector<T> vt = std::vector<T>(m, static_cast<T>(0.));
        Vector<T> vt_ = Vector<T>::Zero(m, 1);
        std::vector<T> vb = std::vector<T>(m, static_cast<T>(0.));
        Vector<T> vb_ = Vector<T>::Zero(m, 1);
        //vi: initial vector for Frank-Wolfe
        //std::vector<T> *vi;
        Vector<T> *vi_;
        std::vector<T> r;
        Vector<T> r_;
        std::vector<T> w(m, static_cast<T>(1.0) / m);
        Vector<T> w_(m);
        w_.setConstant(static_cast<T>(1.0) / m);
        //thresholds for stopping the iteration
        const double eps{0.};
        const double eps_p{1.e-6};
        const double eps1{1.e-4};
        const uint_fast64_t maxIter{20};

        mopmc::optimization::convex_functions::TotalReLU<T> totalReLu1(h);
        mopmc::optimization::convex_functions::TotalReLU<T> totalReLu(h_);
        assert(h.size()== h_.size());


        mopmc::optimization::convex_functions::TotalReLU<T> fn(h_);
        mopmc::optimization::optimizers::FrankWolfe<T> frankWolfe(&fn);

        //Iteration
        uint_fast64_t iter = 0;
        T fDiff1 = 0;
        T fDiff = 0;
        // at least iterate twice
        while (iter < maxIter && (Phi.size() < 3 || fDiff > eps)) {
            std::cout << "Main query loop: Iteration " << iter << "\n";
            if (!Phi.empty()) {
                // compute the FW and find a new weight vector
                //vt = mopmc::solver::convex::frankWolfe(mopmc::solver::convex::reluGradient<T>,
                //                                       *vi, 100, W, Phi, h);
                vt_ = frankWolfe.argmin(Phi_, *vi_, Vertex, true);
                Vector<T> vt1_ = VectorMap<T>(vt.data(), vt.size());
                Vector<T> cx = h_ - vt1_;
                std::vector<T> grad = mopmc::solver::convex::reluGradient(cx);
                Vector<T> grad_ = fn.subgradient(vt_);
                {
                    std::cout << "grad_: [";
                    for (int i = 0; i < m; ++i) {
                        std::cout << grad_(i) << " ";
                    }
                    std::cout << "]\n";
                }
                if (grad_.template lpNorm<1>() < eps_p) {
                    std::cout << "loop exit due to small gradient\n";
                    break;
                }
                /*
                if (mopmc::solver::convex::l1Norm(grad) < eps_p) {
                    break;
                }
                w = mopmc::solver::convex::computeNewW(grad);
                for (double &i: w) {
                    i = -i;
                }
                 */
                w_ = static_cast<T>(-1.) * grad_ / grad_.template lpNorm<1>();
            }
            /*
            std::cout << "w in std vector: ";
            for (double i : w){
                std::cout << i << ",";
            }
            std::cout << "\n";
             */
            std::cout << "w_ in eigen: ";
            for (int i = 0; i < w_.size(); ++i) {
                std::cout << w_(i) << ",";
            }
            std::cout << "\n";


            //GS: As mention, double check whether we need to
            // maintain W and wSet. :SG
            // if the w generated is already contained with W
            /*
            if (wSet.find(w) != wSet.end()) {
                std::cout << "W already in set => W ";
                for (auto val: w) {
                    std::cout << val << ", ";
                }
                std::cout << "\n";
                break;
            }
             */

            // compute a new supporting hyperplane
            std::vector<T> w1(w_.data(), w_.data() + w_.size());
            /*
            std::cout << "w1 before value iteration: ";
            for (double i : w1){
                std::cout << i << ",";
            }
            std::cout << "\n";
             */
            //for (int i=0; i<m; ++i) {assert(w1[i]=w[i]);}
            cudaVIHandler.valueIteration(w1);
            //cudaVIHandler.valueIteration(w);
            // get the first m elements of cudaVIHandler.results_
            r = cudaVIHandler.getResults();
            r.resize(m);
            r_ = VectorMap<T>(r.data(), r.size());
            Phi.push_back(r);
            Phi_.push_back(r_);
            W.push_back(w1);
            W_.push_back(w_);
            wSet.insert(w1);

            //GS: Compute the initial for frank-wolfe and projectedGD.
            // Alright to do it here as the FW function is not called
            // in the first iteration. :SG
            /*
            if (Phi.size() == 1) {
                vi = &r;
            } else {
                vi = &vt;
            }
             */

            if (Phi_.size() == 1) {
                vi_ = &r_;
            } else {
                vi_ = &vt_;
            }

            //T wr = std::inner_product(w.begin(), w.end(), r.begin(), static_cast<T>(0.));
            T wr_ = w_.dot(r_);
           // T wvb = std::inner_product(w.begin(), w.end(), vb.begin(), static_cast<T>(0.));
            T wvb_ = w_.dot(vb_);
            std::vector<T> vv(m);
            Vector<T>::Map(&vv[0], m) = *vi_;
            { //to be removed
                for (int i = 0; i < m; ++i) {
                    assert(vv[i] == (*vi_)(i));
                }
            }
            if (W.size() == 1 || wr_ < wvb_) {
                T gamma = static_cast<T>(0.1);
                vb = mopmc::solver::convex::projectedGradientDescent(
                        mopmc::solver::convex::reluGradient,
                        vv, gamma, 100, Phi, W, Phi.size(), h, eps1);
                vb_ = VectorMap<T>(vb.data(), vb.size());

            }
            //fDiff1 = std::abs(totalReLu1.value1(vt) - totalReLu1.value1(vb));
            fDiff = std::abs(totalReLu.value(vt_) - totalReLu.value(vb_));
            {
                std::cout << "fDiff by eigen vector: " << fDiff << "\n";
                std::cout << "vt_ after fw: [";
                for (int i = 0; i < m; ++i) {
                    std::cout << vt_(i) << " ";
                }
                std::cout << "]\n";
            }
            ++iter;
        }

        cudaVIHandler.exit();
        std::cout << "----------------------------------------------\n";
        std::cout << "@_@ CUDA CONVEX QUERY terminates after " << iter << " iteration(s) \n";
        std::cout << "*Closest point to threshold*: [";
        for (uint_fast64_t i = 0; i < vt_.size(); i ++) {
            std::cout << vt_(i) << " ";
        }
        std::cout << "]\n";
        std::cout << "*Distance to threshold*: " << totalReLu.value(vt_) << "\n";
        std::cout << "----------------------------------------------\n";
    }

    //template class GpuConvexQuery<double, uint64_t>;
    template class GpuConvexQuery<double, int>;
}
