//
// Created by thomas on 5/09/23.
//

#include "ConvexQuery.h"
#include <cassert>
#include <numeric>
#include <iostream>
#include "lp_lib.h"

namespace mopmc::solver::convex {

template <typename T>
T l1Norm(std::vector<T> &x) {
    T acc = static_cast<T>(0.);
    for (T val: x) {
        acc += std::abs(val);
    }
    return acc;
}



template <typename T>
std::vector<T> computeNewW(std::vector<T> &x) {
    // compute the gradient divided by the l1 norm
    std::vector<T> r(x.size());
    T xNorm = l1Norm(x);
    for (uint_fast64_t i = 0; i < x.size() ; ++i) {
        r[i] = x[i] / xNorm;
    }
    return r;
}

template <typename T>
std::vector<T> getS_t(std::vector<T> (*gradient)(Eigen::Matrix<T, Eigen::Dynamic, 1>&),
                      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> &x,
                      std::vector<std::vector<T>> &W,
                      std::vector<std::vector<T>> &Phi,
                      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> &constraints) {
    // compute the function gradient
    T zero_ = static_cast<T>(0.);
    Eigen::Matrix<T, Eigen::Dynamic, 1> cx = constraints - x;
    std::vector<T> grad = gradient(cx);

    // Do the linear programming step
    lprec* problem;
    problem = make_lp(0, x.size());
    // FW is a minimisation problem
    set_minim( problem );
    set_verbose( problem, 3);

    // add a 0 to the beginning of grad
    std::vector<T> grad_ = {zero_};
    grad_.insert(grad_.end(), grad.begin(), grad.end());
    T* c = grad_.data();
    set_obj_fn(problem, c);
    // add the constraints to the problem
    // The constraints are that each w_i.x <= w_i.r_i
    for (uint_fast64_t k = 0; k < W.size(); ++k) {
        // compute the inner product between W_k, Phi_k
        T wr = std::inner_product(W[k].begin(), W[k].end(), Phi[k].begin(), zero_);
        std::vector<T> w_k = {zero_};
        w_k.insert(w_k.end(), W[k].begin(), W[k].end());
        T* wx = w_k.data();

        // TODO this GE depends on whether we are minimising or maximising so we will need
        //  to take it from the environment, just keep it to minimisation at the moment
        add_constraint( problem, wx, GE, wr);
    }
    solve( problem );

    T sol[x.size()];
    get_variables( problem, sol);

    int solSize = sizeof(sol) / sizeof(T);
    std::vector<T> sol_(sol, sol + solSize);
    return sol_;
}

template <typename T>
std::vector<T> frankWolfe(std::vector<T> (*gradient)(Eigen::Matrix<T, Eigen::Dynamic, 1>&),
        std::vector<T>& initialPoint,
        uint64_t numberIterations, std::vector<std::vector<T>> &W,
        std::vector<std::vector<T>>& Phi, std::vector<T>& constraints){
    // set the initial point
    std::vector<T> x = initialPoint, s_t;
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> x_(x.data(), x.size());
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> constraints_(constraints.data(), constraints.size());
    T gamma, gammaConstant = static_cast<T>(2.0);

    // iterate for the set number of iterations or until the error is within some threshold
    for(uint_fast64_t iter = 1; iter < numberIterations + 1; ++iter) {
        gamma = gammaConstant / ( gammaConstant + iter);
        s_t = getS_t(gradient, x_, W, Phi, constraints_);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> s_t_(s_t.data(), s_t.size());;
        x_ += gamma * (s_t_ - x_);
    }
    // return the new point
    std::cout << "x: ";
    for (T val : x) {
        std::cout << val << ",";
    }
    std::cout << "\n";
    return x;
}

template <typename T>
std::vector<T> sub(std::vector<T> &a, std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> c(a.size());
    for(uint_fast64_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] - b[i];
    }
    return c;
}

template<typename T>
T diff(std::vector<T> &fxStar, std::vector<T> &fzStar) {
    std::vector<T> output(fxStar.size());
    T l1result;

    output = sub(fxStar, fzStar);
    l1result = l1Norm(output);
    return l1result;
}

template<typename T>
std::vector<T> scalarMult(T &a, std::vector<T> x) {
    std::vector<T> y(x.size());
    for(uint_fast64_t i = 0; i < x.size(); ++i) {
        y[i] = a * x[i];
    }
    return y;
}

template <typename T>
std::vector<T> unitNormTransformation(std::vector<T> &w) {
    std::vector<T> wUnit(w.size());
    T sumOfSquares = static_cast<T>(0.), denom;
    for(auto val: w) {
        sumOfSquares += val * val;
    }
    denom = std::sqrt(sumOfSquares);
    for (uint_fast64_t i = 0; i < w.size(); ++i) {
        wUnit[i] = w[i] / denom;
    }
    return wUnit;
}

template <typename T>
T computeError(std::vector<T> const& xNew, std::vector<T> const& xOld) {
    T val = static_cast<T>(0.);
    for(uint_fast64_t i = 0; i < xNew.size(); ++i) {
        val += std::abs(xNew[i] - xOld[i]);
    }
    return val;
}

template <typename T>
std::vector<T> projectPointToNearestPlane(std::vector<T> &x,
                                          std::vector<std::vector<T>> &Phi,
                                          std::vector<std::vector<T>> &W,
                                          uint64_t l){
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> dMap;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> dVec;

    T distance = std::numeric_limits<T>::max();
    std::vector<T> xProj(x.size(), 0.);
    dMap xProj_(xProj.data(), xProj.size());
    dMap x_(x.data(), x.size());
    for (uint_fast64_t i = 0; i < l; ++i) {
        // transform w to the unit norm
        std::vector<T> wUnit = unitNormTransformation(W[i]);
        dMap wUnit_(wUnit.data(), wUnit.size());
        dMap r(Phi[i].data(), Phi[i].size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> v = x_ - r;
        T distNew = v.dot(wUnit_);
        std::cout << "distance: " << distNew << "\n";

        if (distNew < distance) {
            distance = distNew;
            xProj_ = x_ - distance * wUnit_;
            std::cout << " xProj: " << xProj_.transpose() << "\n";
        }
    }
    return xProj;
}


template <typename T>
std::vector<T> projectedGradientDescent(
        std::vector<T> (*gradient)(Eigen::Matrix<T, Eigen::Dynamic, 1>&),
        std::vector<T> &initPoint,
        T gamma,
        uint_fast64_t iterations,
        std::vector<std::vector<T>> &Phi,
        std::vector<std::vector<T>> &W,
        uint64_t l,
        std::vector<T> &targetPoint,
        T threshold){
    std::vector<T> x = initPoint;

    for (uint_fast64_t it = 0; it < iterations; ++it){
        std::vector<T> xOld = x;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> x_(x.data(), x.size());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> c_(targetPoint.data(), targetPoint.size());

        Eigen::Matrix<T, Eigen::Dynamic, 1> cx = x_ - c_;
        std::vector<T> grad = gradient(cx);

        for (uint_fast64_t i = 0; i < x.size() ; ++i) {
            x[i] = x[i] - gamma * grad[i];
        }

        std::vector<T> xNew = projectPointToNearestPlane(x, Phi, W, l);
        // compute the error between xnew and xold to determine convergence
        T error = computeError(xNew, xOld);
        std::cout << "error: " << error << "\n";
        if (error < threshold) {
            x = xNew;
            break;
        }
        x = xNew;
    }
    return x;
}

// Explicit Instantiation
template std::vector<double> frankWolfe(std::vector<double> (*gradient)(Eigen::Matrix<double, Eigen::Dynamic, 1>&),
                std::vector<double>& initialPoint, uint64_t numberIterations,
                std::vector<std::vector<double>>& weightVector,
                std::vector<std::vector<double>>& Phi, std::vector<double>& constraints);

//template std::vector<double> getS_t(std::vector<double> (*gradient)(std::vector<double>), std::vector<double> &x, std::vector<std::vector<double>> &W, std::vector<std::vector<double>> &Phi,
//                      std::vector<double> &constraints);

//std::vector<double> reluGradient(std::vector<double> x);

template std::vector<double> sub(std::vector<double> &a, std::vector<double> &b);

template std::vector<double> computeNewW(std::vector<double> &x);

template std::vector<double> projectedGradientDescent(
        std::vector<double> (*gradient)(Eigen::Matrix<double, Eigen::Dynamic, 1>&),
        std::vector<double> &initPoint,
        double gamma,
        uint_fast64_t iterations,
        std::vector<std::vector<double>> &Phi,
        std::vector<std::vector<double>> &W,
        uint64_t l,
        std::vector<double> &targetPoint,
        double threshold);

template double l1Norm(std::vector<double> &x);

template double diff(std::vector<double> &fxStar, std::vector<double> &fzStar);

}
