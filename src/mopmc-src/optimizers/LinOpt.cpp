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
        } else if (rep == Halfspace) {
            assert(!W.empty());
            Ncol = W[0].size();
        } else {
            throw std::runtime_error("this linopt method does not support this polytope representation type");
        }

        /* We start with creating a model with 0 rows and @Ncol columns */
        lp = make_lp(0, Ncol);

        colnoVec.resize(Ncol);
        rowVec.resize(Ncol);
        if (lp == NULL)
            ret = 1; /* couldn't construct a new model... */

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


        if (ret == 0) {
            set_add_rowmode(lp, TRUE);  /* makes building the model faster if it is done rows by row */
            // constraints
            for (int i = 0; i < Phi.size(); ++i) {
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
            if (!add_constraintex(lp, Ncol - 1, rowVec.data(), colnoVec.data(), EQ, static_cast<V>(1.)))
                ret = 3;
            for (int j = 0; j < Ncol - 1; ++j) {
                colnoVec[j] = j + 1;
                rowVec[j] = static_cast<V>(1.);
                if (!add_constraintex(lp, 1, rowVec.data(), colnoVec.data(), GE, static_cast<V>(0.)))
                    ret = 3;
            }
            set_add_rowmode(lp, FALSE);
        }

        if (ret == 0) {
            /* set the object direction to maximize */
            set_maxim(lp);
            /* just out of curiosity, now show the model in lp format on screen */
            /* this only works if this is a console application. If not, use write_lp and a filename */
            //write_LP(lp, stdout); /* write_lp(lp, "model.lp"); */
            /* only to see important messages on screen while solving */
            set_verbose(lp, IMPORTANT);
            /* Now let lpsolve calculate a solution */
            if (solve(lp) != OPTIMAL)
                ret = 5;

        }
        if (ret == 0) {
            /* a solution is calculated, now lets get some results */
            /* objective value */
            //printf("Objective value1: %f\n", get_objective(lp));
            /* variable values */
            get_variables(lp, rowVec.data());
            for (int j = 0; j < Ncol; ++j)
                printf("%s: %f\n", get_col_name(lp, j + 1), rowVec[j]);
            /* we are done now */
        }
        //std::cout << "LINOPT: GOT HERE with ret = " << ret <<"\n";

        /* return result in each case */
        if (rep == Vertex) {
            optValues.setZero();
            for (int j = 0; j < Ncol; ++j) {
                optValues += rowVec[j] * Phi[j];
            }
        }
        if (rep == Halfspace) {
            optValues = VectorMap<V>(rowVec.data(), Ncol);
        }

        if (lp != NULL) { delete_lp(lp); }
        return ret;
    };

    template<typename V>
    int LinOpt<V>::optimizeInFW(std::vector<Vector<V>> &Phi,
                                std::vector<Vector<V>> &W,
                                PolytopeType &rep,
                                Vector<V> &d,
                                Vector<V> &optValues) {

        assert(rep == Halfspace);
        Vector<V> sgn1 = Vector<V>::Ones(d.size());
        return argmin(Phi, W, rep, d, sgn1, optValues);
    }

    template<typename V>
    int LinOpt<V>::optimizeInFW(std::vector<Vector<V>> &Phi,
                                PolytopeType &rep,
                                Vector<V> &d,
                                Vector<V> &optValues) {

        assert(rep == Vertex);
        Vector<V> sgn1 = Vector<V>::Ones(d.size());
        std::vector<Vector<V>> W0;
        return argmin(Phi, W0, rep, d, sgn1, optValues);
    }

    template<typename V>
    int LinOpt<V>::optimize(std::vector<Vector<V>> &Phi,
                            PolytopeType &rep,
                            Vector<V> &d,
                            Vector<V> &sgn,
                            Vector<V> &out) {

        assert(rep == Closure);
        lprec *lp;
        int n_cols, *col_no = NULL, ret = 0;
        V *row = NULL;

        assert(!Phi.empty());
        n_cols = Phi[0].size() + 1; // number of variables in the model
        lp = make_lp(0, n_cols);
        if (lp == NULL)
            ret = 1; // couldn't construct a new model
        if (ret == 0) {
            //[important!] set the unbounded variables.
            // The default bounds are >=0 in lp solve.
            set_unbounded(lp, n_cols);
            // create space large enough for one row
            col_no = (int *) malloc(n_cols * sizeof(*col_no));
            row = (V *) malloc(n_cols * sizeof(*row));
            if ((col_no == NULL) || (row == NULL))
                ret = 2;
        }

        if (ret == 0) {
            set_add_rowmode(lp, TRUE);
            // constraints
            for (int j = 0; j < n_cols - 1; ++j) {
                col_no[j] = j + 1;
                row[j] = static_cast<V>(1.);
            }
            if (!add_constraintex(lp, n_cols - 1, row, col_no, GE, static_cast<V>(1.)))
                ret = 3;
            if (!add_constraintex(lp, n_cols - 1, row, col_no, LE, static_cast<V>(1.)))
                ret = 3;
            for (int j = 0; j < n_cols - 1; ++j) {
                col_no[0] = j + 1;
                row[0] = static_cast<V>(1.);
                if (!add_constraintex(lp, 1, row, col_no, GE, static_cast<V>(0.)))
                    ret = 3;
            }
            for (int i = 0; i < Phi.size(); ++i) {
                for (int j = 0; j < n_cols - 1; ++j) {
                    col_no[j] = j + 1;
                    row[j] = sgn(j) * (d(j) - Phi[i](j));
                    //std::cout << "row[j]: " << row[j] <<"\n";
                }
                col_no[n_cols - 1] = n_cols;
                row[n_cols - 1] = static_cast<V>(-1.);
                if (!add_constraintex(lp, n_cols, row, col_no, GE, static_cast<V>(0.)))
                    ret = 3;
            }
        }

        if (ret == 0) {
            set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model
            col_no[0] = n_cols;
            row[0] = static_cast<V>(1.);/* set the objective in lpsolve */
            if (!set_obj_fnex(lp, 1, row, col_no))
                ret = 4;
        }

        if (ret == 0) {
            // set the object direction to maximize
            set_maxim(lp);
            //write_LP(lp, stdout);
            // write_lp(lp, "model.lp");
            set_verbose(lp, 3);
            ret = solve(lp);
            //std::cout<< "** Optimal solution? Ret: " << ret << "\n";
            if (ret == OPTIMAL)
                ret = 0;
            else
                ret = 5;
        }

        if (ret == 0) {
            get_variables(lp, row);
            std::cout << "Optimal solutions: ";
            for (int j = 0; j < n_cols; j++)
                std::cout << get_col_name(lp, j + 1) << ": " << row[j] << ", ";
            std::cout << "\n";
            // we are done now
        }

        out = VectorMap<V>(row, n_cols);
        // free allocated memory
        if (row != NULL)
            free(row);
        if (col_no != NULL)
            free(col_no);
        if (lp != NULL) {
            // clean up such that all used memory by lpsolve is freed
            delete_lp(lp);
        }

        return ret;
    };

    template
    class LinOpt<double>;
}