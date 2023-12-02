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

        std::vector<std::vector<T>> rho(m);
        std::vector<T> rho_flat(n * m);//rho: all reward vectors
        //GS: need to store whether an objective is probabilistic or reward-based.
        //TODO In future we will use treat them differently in the loss function. :GS
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
        std::vector<T> *vi;
        Vector<T> *vi_;
        std::vector<T> r;
        Vector<T> r_;
        std::vector<T> w = {-.0, -1.};// w: weight vector
                // std::vector<T>(m, static_cast<T>(-1.0) / static_cast<T>(m));
        Vector<T> w_ = VectorMap<T> (w.data(), w.size());
        //thresholds for stopping the iteration
        const double eps{0.};
        const double eps_p{1.e-6};
        const double eps1{1.e-4};
        const uint_fast64_t maxIter{20};

        mopmc::optimization::convex_functions::TotalReLU<T> totalReLu1(h);
        mopmc::optimization::convex_functions::TotalReLU<T> totalReLu(h_);
        assert(h.size()== h_.size());

        //Iteration
        uint_fast64_t iter = 0;
        T fDiff1 = 0;
        T fDiff = 0;
        while (iter < maxIter && (Phi.size() < 3 || fDiff1 > eps)) {
            std::cout << "Iteration: " << iter << "\n";
            //std::vector<T> fvt = mopmc::solver::convex::ReLU(vt, h);
            //std::vector<T> fvb = mopmc::solver::convex::ReLU(vb, h);
            //fDiff = mopmc::solver::convex::diff(fvt, fvb);
            fDiff1 = std::abs(totalReLu1.value1(vt) - totalReLu1.value1(vb));
            fDiff = std::abs(totalReLu.value(vt_) - totalReLu.value(vb_));
            std::cout << "fDiff1 by std vector: " << fDiff1 << ", fDiff by eigen vector: " << fDiff <<"\n";
            std::cout << "vt by std vector: [" << vt[0] <<", " << vt[1] <<"]\n";
            std::cout << "vt_ by eigen: [" << vt_(0) <<", " << vt_(1) <<"]\n";
            mopmc::optimization::convex_functions::TotalReLU<T> fn(h_);
            mopmc::optimization::optimizers::FrankWolfe<T> frankWolfe(&fn);
            //assert(fDiff1==fDiff);
            if (!Phi.empty()) {
                // compute the FW and find a new weight vector
                vt = mopmc::solver::convex::frankWolfe(mopmc::solver::convex::reluGradient<T>,
                                                       *vi, 100, W, Phi, h);
                vt_ = frankWolfe.argmin(Phi_, *vi_, Vertex, false);

                Vector<T> vt1_ = VectorMap<T>(vt.data(), vt.size());
                //GS: To be consistent, may change the arg type of
                // reluGradient() to vector. :GS
                //Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vt_(vt.data(), vt.size());
                Vector<T> cx = h_ - vt1_;
                std::vector<T> grad = mopmc::solver::convex::reluGradient(cx);
                Vector<T> grad_ = fn.subgradient(vt_);
                //GS: exit if the gradient is very small. :SG
                //std::cout << "L1 norm of gradient by std vector: " << mopmc::solver::convex::l1Norm(grad) <<"\n";
                if (mopmc::solver::convex::l1Norm(grad) < eps_p) {
                    break;
                }
                w = mopmc::solver::convex::computeNewW(grad);
                for (double & i : w) {
                 i = -i;
                }

                //if (grad_.template lpNorm<1>() < eps_p ) { break; }
                //assert (grad_.size() == 2);
                //std::cout << "grad_(0): " << grad_(0) <<std::endl;
                //std::cout << "grad_(1): " << grad_(1) <<std::endl;
                //std::cout << "grad_.template lpNorm<1>(): " << grad_.template lpNorm<1>() <<std::endl;
                w_ = static_cast<T>(-1.) * grad_ / grad_.template lpNorm<1>();
            }
            /*
            std::cout << "w in std vector: ";
            for (double i : w){
                std::cout << i << ",";
            }
            std::cout << "\n";
            std::cout << "w_ in eigen: ";
            for (int i=0; i<w_.size(); ++i){
                std::cout << w_(i) << ",";
            }
            std::cout << "\n";
             */


            //GS: As mention, double check whether we need to
            // maintain W and wSet. :SG
            // if the w generated is already contained with W

            if (wSet.find(w) != wSet.end()) {
                std::cout << "W already in set => W ";
                for (auto val: w) {
                    std::cout << val << ", ";
                }
                std::cout << "\n";
                break;
            }

            // compute a new supporting hyperplane
            std::vector<T> w1(w_.data(), w_.data() + w_.size());
            /*
            std::cout << "w1 before value iteration: ";
            for (double i : w1){
                std::cout << i << ",";
            }
            std::cout << "\n";
             */
            //cudaVIHandler.valueIteration(w1);
            cudaVIHandler.valueIteration(w);
            // get the first m elements of cudaVIHandler.results_
            r = cudaVIHandler.getResults();
            r.resize(m);
            r_ = VectorMap<T>(r.data(), r.size());
            //std::cout << "HEHE!!!\n";
            Phi.push_back(r);
            Phi_.push_back(r_);
            //LambdaL.push_back(w);
            //LambdaR.push_back(r);
            W.push_back(w);
            W_.push_back(w_);
            wSet.insert(w);

            //GS: Compute the initial for frank-wolfe and projectedGD.
            // Alright to do it here as the FW function is not called
            // in the first iteration. :SG
            if (Phi.size() == 1) {
                vi = &r;
            } else {
                vi = &vt;
            }

            if (Phi_.size() == 1) {
                vi_ = &r_;
            } else {
                vi_ = &vt_;
            }

            T wr = std::inner_product(w.begin(), w.end(), r.begin(), static_cast<T>(0.));
            T wr_ = w_.dot(r_);
            T wvb = std::inner_product(w.begin(), w.end(), vb.begin(), static_cast<T>(0.));
            T wvb_ = w_.dot(vb_);
            if (W.size() == 1 || wr < wvb) {
                T gamma = static_cast<T>(0.1);
                std::cout << "|Phi|: " << Phi.size() << "\n";
                vb = mopmc::solver::convex::projectedGradientDescent(
                        mopmc::solver::convex::reluGradient,
                        *vi, gamma, 10, Phi, W, Phi.size(), h, eps1);
                vb_ = frankWolfe.argmin(Phi_, W_, *vi_, Halfspace, false);
            }
            ++iter;
        }

        cudaVIHandler.exit();
        std::cout << "----------------------------------------------\n";
        std::cout << "@_@ CUDA CONVEX QUERY terminates after " << iter << " iteration(s) \n";
        std::cout << "*Distance to thresholds*: " << totalReLu.value(vt_) << "\n";
        std::cout << "Result at initial state ";
        for (int i = 0; i < this->data_.objectiveCount; ++i) {
            std::cout << "- Objective " << i << ": " << cudaVIHandler.getResults()[i] << " ";
        }std::cout <<"\n";
        std::cout << "(Negative) Weighted result: " << cudaVIHandler.getResults()[this->data_.objectiveCount] << "\n";
        std::cout << "----------------------------------------------\n";
    }

    //template class GpuConvexQuery<double, uint64_t>;
    template class GpuConvexQuery<double, int>;
}
