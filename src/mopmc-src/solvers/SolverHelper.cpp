//
// Created by thomas on 26/09/23.
//

#include "SolverHelper.h"
#include <storm/models/sparse/Mdp.h>

namespace mopmc::solver::helper {
    template<typename SparseModelType>
    Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> eigenInducedTransitionMatrix(
            storm::storage::SparseMatrix<typename SparseModelType::ValueType> &fullTransitionSystem,
            std::vector<uint64_t> &scheduler,
            std::vector<uint_fast64_t> const &rowGroupIndices
    ) {

        assert(scheduler.size() == fullTransitionSystem.getColumnCount());
        Eigen::SparseMatrix<typename SparseModelType::ValueType, Eigen::RowMajor> subMatrix(scheduler.size(),
                                                                                            scheduler.size());
        for (uint_fast64_t state = 0; state < fullTransitionSystem.getColumnCount(); ++state) {
            auto const &actionStart = fullTransitionSystem.getRowGroupIndices()[state];
            for (auto element : fullTransitionSystem.getRow(actionStart)) {
                subMatrix.insert(state, element.getColumn()) = element.getValue();
            }
        }
        subMatrix.makeCompressed();
        return subMatrix;
    }

    template<typename ValueType>
    void inducedRewards(std::vector<ValueType> &b, std::vector<ValueType> &subB,
                        std::vector<uint64_t> &scheduler, std::vector<uint_fast64_t> const &rowGroupIndices) {
        for (uint_fast64_t state = 0; state < scheduler.size(); ++state) {
            auto const &actionStart = rowGroupIndices[state];
            //std::cout << "state " << state << " action " << actionStart;
            subB[state] = b[actionStart + scheduler[state]];
        }
    }

    template Eigen::SparseMatrix<double, Eigen::RowMajor>
    eigenInducedTransitionMatrix<storm::models::sparse::Mdp<double>>(
            storm::storage::SparseMatrix<double> &fullTransitionSystem,
            std::vector<uint64_t> &scheduler,
            std::vector<uint_fast64_t> const &rowGroupIndices);

    template void inducedRewards(std::vector<double> &b, std::vector<double> &subB,
                                      std::vector<uint64_t> &scheduler,
                                      std::vector<uint_fast64_t> const &rowGroupIndices);
}