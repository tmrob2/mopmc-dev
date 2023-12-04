//
// Created by guoxin on 4/12/23.
//




#include "GpuConvexQueryAlt.h"
#include <iostream>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "../solvers/ConvexQuery.h"
#include "../solvers/CudaValueIteration.cuh"
#include "../Data.h"
#include "../convex-functions/TotalReLU.h"
#include "../convex-functions/EuclideanDistance.h"
#include "../convex-functions/SignedKLEuclidean.h"
#include "../optimizers/FrankWolfe.h"
#include "../optimizers/PolytopeTypeEnum.h"

namespace mopmc::queries {

    template<typename T, typename I>
    void GpuConvexQueryAlt<T, I>::query() {

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
        const uint64_t k = this->data_.colCount; // k: number of states
        assert(this->data_.rowGroupIndices.size()==k+1);
        std::vector<T> h = this->data_.thresholds;
        Vector<T> h_ = Eigen::Map<Vector<T>>(this->data_.thresholds.data(), this->data_.thresholds.size());

        std::vector<std::vector<T>> Phi;
        std::vector<Vector<T>> Phi_;
        std::vector<std::vector<T>> W;
        std::vector<Vector<T>> W_;
        std::set<std::vector<T>> wSet;
        std::vector<T> vt = std::vector<T>(m, static_cast<T>(0.));
        Vector<T> vt_ = Vector<T>::Zero(m, 1);
        std::vector<T> vb = std::vector<T>(m, static_cast<T>(0.));
        Vector<T> vb_ = Vector<T>::Zero(m, 1);
        Vector<T> *vi_;
        std::vector<T> r;
        Vector<T> r_;
        Vector<T> rTemp_;
        std::vector<T> w(m, static_cast<T>(1.0) / m);
        Vector<T> w_(m);
        w_.setConstant(static_cast<T>(1.0) / m);

        //const double eps{0.};
        const double eps_p{1.e-6};
        //const double eps1{1.e-4};
        const uint_fast64_t maxIter{20};

        mopmc::optimization::convex_functions::TotalReLU<T> totalReLu(h_);
        assert(h.size()== h_.size());

        mopmc::optimization::convex_functions::EuclideanDistance<T> fn(h_);
        mopmc::optimization::optimizers::FrankWolfe<T> frankWolfe(&fn);

        //Iteration
        uint_fast64_t iter = 0;
        //T fDiff1 = 0;
        //T fDiff = 0;
        // at least iterate twice
        while (iter < maxIter) {
            std::cout << "Main query loop: Iteration " << iter << "\n";
            if (!Phi.empty()) {
                vt_ = frankWolfe.argmin(Phi_, *vi_, Vertex, false);
                Vector<T> vt1_ = VectorMap<T>(vt.data(), vt.size());
                Vector<T> grad_ = fn.subgradient(vt_);
                std::cout << "Distance to threshold: " << totalReLu.value(vt_) << "\n";
                /*{
                    std::cout << "grad_: [";
                    for (int i = 0; i < m; ++i) {
                        std::cout << grad_(i) << " ";
                    }
                    std::cout << "]\n";
                }*/
                if (grad_.template lpNorm<1>() < eps_p) {
                    std::cout << "loop exit due to small gradient\n";
                    break;
                }
                w_ = static_cast<T>(-1.) * grad_ / grad_.template lpNorm<1>();
            }
            /*
            std::cout << "w in std vector: ";
            for (double i : w){
                std::cout << i << ",";
            }
            std::cout << "\n";
             */
            /*{
                std::cout << "w_ in eigen: ";
                for (int i = 0; i < w_.size(); ++i) {
                    std::cout << w_(i) << ",";
                }
                std::cout << "\n";
            }*/

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

            if (!Phi.empty()) {
                mopmc::optimization::convex_functions::EuclideanDistance<T> fn1(r_);
                mopmc::optimization::optimizers::FrankWolfe<T> frankWolfe1(&fn1);
                /*
                rTemp_ = frankWolfe.argmin(Phi_, *vi_, Vertex, false);
                if ((rTemp_ - r_).template lpNorm<1>() < 0.001) {
                    //std::cout << "loop exit due to ...\n";
                    break;
                }
                 */
                /*
                bool b = false;
                for (uint_fast64_t j = 0; j < Phi_.size(); ++j) {
                    if ((Phi_[j] - r_).template lpNorm<1>() < 1.e-8) {
                        b = true;
                    }
                    //std::cout << "(Phi_[" << j << "] - r_).template lpNorm<1>(): " << (Phi_[j] - r_).template lpNorm<1>() <<"\n";
                }
                if (b) {
                    //std::cout << "loop exit due to ...\n";
                    //break;
                }
                 */
                {
                    /*
                    for (int i = 0; i < m; ++i) {
                        std::cout << rTemp_(i) << " ";
                    }
                    std::cout << "]\n";
                     */
                    /*
                    std::cout << "r_: [";
                    for (int i = 0; i < m; ++i) {
                        std::cout << r_(i) << " ";
                    }
                    std::cout << "]\n";
                    */
                }
                //std::cout << "(rTemp_ - r_).template lpNorm<1>(): " << (rTemp_ - r_).template lpNorm<1>() <<"\n";
            }

            Phi.push_back(r);
            Phi_.push_back(r_);
            W.push_back(w1);
            W_.push_back(w_);
            wSet.insert(w1);

            /*{
                std::cout << "vt_: [";
                for (int i = 0; i < m; ++i) {
                    std::cout << vt_(i) << " ";
                }
                std::cout << "]\n";
            }*/

            if (Phi_.size() == 1) {
                vi_ = &r_;
            } else {
                vi_ = &vt_;
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
    template class GpuConvexQueryAlt<double, int>;
}