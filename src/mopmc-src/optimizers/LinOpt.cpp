//
// Created by guoxin on 24/11/23.
//

#include <iostream>
#include "LinOpt.h"

namespace mopmc::optimization::optimizers {

    template<typename V>
    int LinOpt<V>::argmin(std::vector<Vector<V>> &Phi, std::vector<Vector<V>> &W, PolytopeType &rep, Vector<V> &d,
                          Vector<V> &sgn, Vector<V> &optValues) {
        //Reference https://lpsolve.sourceforge.net/5.5/
        lprec *lp;
        int Ncol;
        //int *colno = NULL;
        int ret = 0;
        //V *row = NULL;
        std::vector<int> colnoVec;
        std::vector<V> rowVec;

        assert(!Phi.empty());
        /* Define number of variables in each case */
        if (rep == Vertex) {
            Ncol = Phi.size();
        }
        if (rep == Halfspace) {
            assert(!W.empty());
            Ncol = W[0].size();
        }
        if (rep == Closure) {
            Ncol = Phi[0].size() + 1;
        }

        std::cout << "LINOPT: GOT HERE with ret = " << ret <<"\n";

        /* We start with creating a model with 0 rows and @Ncol columns */
        lp = make_lp(0, Ncol);

        colnoVec.reserve(Ncol);
        rowVec.reserve(Ncol);
        if (lp == NULL)
            ret = 1; /* couldn't construct a new model... */
        if (ret == 0) {
            /* let us name our variables. Not required, but can be useful for debugging */
            //for (int j=0; j<Ncol; ++j) {
            //    set_col_name(lp, j+1, std::(j));
            //}
        }

        /* We will build the model row by row, depending on each case */
        if (rep == Vertex) {
            if (ret == 0) {
                Vector<V> c(Ncol);
                for (int j = 0; j < Ncol; ++j) {
                    Vector<V> r = Phi[j];
                    assert(r.size() == d.size());
                    c(j) = d.dot(r);
                    colnoVec[j] = j + 1;
                    rowVec[j] = c(j);
                }
                /* set the objective in lpsolve */
                if (!set_obj_fnex(lp, Ncol, rowVec.data(), colnoVec.data()))
                    ret = 4;
            }
            if (ret == 0) {
                set_add_rowmode(lp, TRUE);  /* makes building the model faster if it is done rows by row */
                // first constraint
                for (int j = 0; j < Ncol; ++j) {
                    colnoVec[j] = j + 1;
                    rowVec[j] = static_cast<V>(1.);
                }
                /* add the row to lpsolve */
                if (!add_constraintex(lp, Ncol, rowVec.data(), colnoVec.data(), LE, static_cast<V>(1.)))
                    ret = 3;
                // other constraints
                for (int j = 0; j < Ncol; ++j) {
                    colnoVec[0] = j + 1;
                    rowVec[0] = static_cast<V>(1.);
                    if (!add_constraintex(lp, 1, rowVec.data(), colnoVec.data(), GE, static_cast<V>(0.)))
                        ret = 3;
                }
                set_add_rowmode(lp, FALSE);
            }
        }

        if (rep == Halfspace) {
            if (ret == 0) {
                for (int j = 0; j < Ncol; ++j) {
                    colnoVec[j] = j + 1;
                    rowVec[j] = d(j);
                }
                /* set the objective in lpsolve */
                if (!set_obj_fnex(lp, Ncol, rowVec.data(), colnoVec.data()))
                    ret = 4;
            }
            if (ret == 0) {
                set_add_rowmode(lp, TRUE);  /* makes building the model faster if it is done rows by row */
                // constraints
                for (int i = 0; i < Phi.size(); ++i) {
                    for (int j = 0; j < Ncol; ++j) {
                        colnoVec[0] = j + 1;
                        rowVec[0] = W[i](j);
                    }
                    V wr = W[i].dot(Phi[i]);
                    if (!add_constraintex(lp, Ncol, rowVec.data(), colnoVec.data(), LE, wr))
                        ret = 3;
                }
                set_add_rowmode(lp, FALSE);
            }
        }

        if (rep == Closure) {
            if (ret == 0) {
                colnoVec[0] = Ncol;
                rowVec[0] = static_cast<V>(1.);
                /* set the objective in lpsolve */
                if (!set_obj_fnex(lp, Ncol, rowVec.data(), colnoVec.data()))
                    ret = 4;
            }

            std::cout << "LINOPT: GOT HERE with ret = " << ret <<"\n";
        }
        if (ret == 0) {
            set_add_rowmode(lp, TRUE);  /* makes building the model faster if it is done rows by row */
            // constraints
            for (int i = 0; i < Phi.size(); ++i ) {
                for (int j = 0; j < Ncol - 1; ++j) {
                    //colno[j] = j + 1;
                    colnoVec[j] = j + 1;
                    rowVec[j] = sgn(j) * (d(j) - Phi[i](j));
                }
                colnoVec[Ncol] = Ncol;
                rowVec[Ncol] = static_cast<V>(-1.);
                if (!add_constraintex(lp, Ncol, rowVec.data(), colnoVec.data(), GE, static_cast<V>(0.)))
                    ret = 3;
            }
            for (int j = 0; j < Ncol - 1; ++j) {
                colnoVec[j] = j + 1;
                rowVec[j] = static_cast<V>(1.);
            }
            if (!add_constraintex(lp, Ncol-1, rowVec.data(), colnoVec.data(), EQ, static_cast<V>(1.)))
                ret = 3;
            for (int j = 0; j < Ncol - 1; ++j) {
                colnoVec[j] = j + 1;
                rowVec[j] = static_cast<V>(1.);
                if (!add_constraintex(lp, 1, rowVec.data(), colnoVec.data(), GE, static_cast<V>(0.)))
                    ret = 3;
            }
            set_add_rowmode(lp, FALSE);
            std::cout << "LINOPT: GOT HERE\n";
        }

        if (ret == 0) {
            /* set the object direction to maximize */
            set_maxim(lp);
            /* just out of curiosity, now show the model in lp format on screen */
            /* this only works if this is a console application. If not, use write_lp and a filename */
            write_LP(lp, stdout); /* write_lp(lp, "model.lp"); */
            /* only to see important messages on screen while solving */
            set_verbose(lp, IMPORTANT);
            /* Now let lpsolve calculate a solution */
            if (solve(lp) != OPTIMAL)
                ret = 5;

            std::cout << "LINOPT: GOT HERE with ret = " << ret <<"\n";

        }
        if (ret == 0) {
            /* a solution is calculated, now lets get some results */
            /* objective value */
            //printf("Objective value1: %f\n", get_objective(lp));
            /* variable values */
            get_variables(lp, rowVec.data());
            //for(int j = 0; j < Ncol; ++j)
            //   printf("%s: %f\n", get_col_name(lp, j + 1), row[j]);
            /* we are done now */
        }
        std::cout << "LINOPT: GOT HERE with ret = " << ret <<"\n";

        /* return result in each case */
        if (rep == Vertex) {
            optValues.setZero();
            for (int j = 0; j < Ncol; ++j ){
                optValues += rowVec[j] * Phi[j];
            }
        }
        if (rep == Halfspace){
            optValues = VectorMap<V> (rowVec.data(), Ncol);
        }
        if (rep == Closure) {
            optValues = VectorMap<V> (rowVec.data(), Ncol);

        }

        if (lp != NULL) { delete_lp(lp); }
        return ret;
    };

    template<typename V>
    int LinOpt<V>::argmin(std::vector<Vector<V>> &Phi,
                          std::vector<Vector<V>> &W,
                          PolytopeType &rep,
                          Vector<V> &d,
                          Vector<V> &optValues) {

        assert(rep == Halfspace);
        Vector<V> sgn1 = Vector<V>::Ones(d.size());
        return argmin(Phi, W, rep, d, sgn1, optValues);
    }

    template<typename V>
    int LinOpt<V>::argmin(std::vector<Vector<V>> &Phi,
                          PolytopeType &rep,
                          Vector<V> &d,
                          Vector<V> &optValues) {

        assert(rep == Vertex);
        Vector<V> sgn1 = Vector<V>::Ones(d.size());
        std::vector<Vector<V>> W0;
        return argmin(Phi, W0, rep, d, sgn1, optValues);
    }

    template<typename V>
    int LinOpt<V>::argmin(std::vector<Vector<V>> &Phi,
               PolytopeType &rep,
               Vector<V> &d,
               Vector<V> &sgn,
               Vector<V> &optValues) {

        assert(rep == Closure);
        std::vector<Vector<V>> W0;
        return argmin(Phi, W0, rep, d, sgn, optValues);
    };

    template
    class LinOpt<double>;
}