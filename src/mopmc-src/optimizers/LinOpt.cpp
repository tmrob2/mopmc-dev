//
// Created by guoxin on 24/11/23.
//

#include "LinOpt.h"

namespace mopmc::optimization::optimizers {
    template<typename V>
    int LinOpt<V>::argmin(std::vector<std::vector<V>> &Phi, std::vector<std::vector<V>> &W, PolytopeRep &rep,
                          std::vector<V> d, std::vector<V> &optValues) {
        //Reference https://lpsolve.sourceforge.net/5.5/
        lprec *lp;
        int Ncol, *colno = NULL, ret = 0;
        V *row = NULL;
        if (rep == VRep) {
            /* We will build the model row by row
            So we start with creating a model with 0 rows and @Ncol columns */
            Ncol = Phi.size(); /* there are @Phi.size() variables in the model */
            lp = make_lp(0, Ncol);
            if (lp == NULL) ret = 1; /* couldn't construct a new model... */
            if (ret == 0) {
                /* let us name our variables. Not required, but can be useful for debugging */
                //for (int j=0; j<Ncol; ++j) {
                //    set_col_name(lp, j+1, std::(j));
                //}
                colno = (int *) malloc(Ncol * sizeof(*colno));
                row = (V *) malloc(Ncol * sizeof(*row));
                if ((colno == NULL) || (row == NULL)) {
                    ret = 2;
                }
            }
            if (ret == 0) {
                std::vector<V> c(Ncol);
                for (int j = 0; j < Ncol; ++j) {
                    std::vector<V> r = Phi[j];
                    assert(r.size() == d.size());
                    c[j] = std::inner_product(d.begin(), d.end(), r.begin(), static_cast<V>(0.));
                    colno[j] = j + 1;
                    row[j] = c[j];
                }
                /* set the objective in lpsolve */
                if (!set_obj_fnex(lp, Ncol, row, colno))
                    ret = 4;
            }
            if (ret == 0) {
                set_add_rowmode(lp, TRUE);  /* makes building the model faster if it is done rows by row */
                // first constraint
                for (int j = 0; j < Ncol; ++j) {
                    colno[j] = j + 1;
                    row[j] = static_cast<V>(1.);
                }
                /* add the row to lpsolve */
                if (!add_constraintex(lp, Ncol, row, colno, LE, static_cast<V>(1.)))
                    ret = 3;;
                // other constraints
                for (int j = 0; j < Ncol; ++j) {
                    colno[0] = j + 1;
                    row[0] = static_cast<V>(1.);
                    if (!add_constraintex(lp, 1, row, colno, GE, static_cast<V>(0.)))
                        ret = 3;
                }
                set_add_rowmode(lp, FALSE);
            }
        } else {
            assert (!W.empty());

            Ncol = W[0].size();
            if (ret == 0) {
                colno = (int *) malloc(Ncol * sizeof(*colno));
                row = (V *) malloc(Ncol * sizeof(*row));
                if ((colno == NULL) || (row == NULL)) {
                    ret = 2;
                }
            }
            if (ret == 0) {
                for (int j = 0; j < Ncol; ++j) {
                    colno[j] = j + 1;
                    row[j] = d[j];
                }
                /* set the objective in lpsolve */
                if (!set_obj_fnex(lp, Ncol, row, colno))
                    ret = 4;
            }
            if (ret == 0) {
                set_add_rowmode(lp, TRUE);  /* makes building the model faster if it is done rows by row */
                // constraints
                //std::vector<V> c(Ncol);
                for (int i = 0; i < Phi.size(); ++i) {
                    for (int j = 0; j < Ncol; ++j) {
                        colno[0] = j + 1;
                        row[0] = W[i][j];
                    }
                    V e = std::inner_product(W[i].begin(), W[i].end(), Phi[i].begin(), static_cast<V>(0.));
                    if (!add_constraintex(lp, Ncol, row, colno, LE, e))
                        ret = 3;
                }
                set_add_rowmode(lp, FALSE);
            }
        }

        if (ret == 0) {
            /* set the object direction to maximize */
            set_minim(lp);
            /* just out of curioucity, now show the model in lp format on screen */
            /* this only works if this is a console application. If not, use write_lp and a filename */
            write_LP(lp, stdout); /* write_lp(lp, "model.lp"); */
            /* only to see important messages on screen while solving */
            set_verbose(lp, IMPORTANT);
            /* Now let lpsolve calculate a solution */
            if (solve(lp) != OPTIMAL)
                ret = 5;
        }
        if (ret == 0) {
            /* a solution is calculated, now lets get some results */
            /* objective value */
            printf("Objective value: %f\n", get_objective(lp));
            /* variable values */
            get_variables(lp, row);
            //for(int j = 0; j < Ncol; j++)
            //    printf("%s: %f\n", get_col_name(lp, j + 1), row[j]);
            /* we are done now */
        }
        /* free allocated memory */
        if (row != NULL) { free(row); }
        if (colno != NULL) { free(colno); }
        if (lp != NULL) { delete_lp(lp); }

        return (ret);
    }


    template<typename V>
    int LinOpt<V>::argmin(std::vector<std::vector<V>> &Phi, PolytopeRep &rep, std::vector<V> d,
                          std::vector<V> &optValues) {
        std::vector<std::vector<V>> W0 = std::vector<std::vector<V>>();
        return argmin(Phi, W0, rep, d, optValues);
    }

    template
    class LinOpt<double>;
}