//
// Created by guoxin on 2/11/23.
//

#ifndef MOPMC_RUNNER_H
#define MOPMC_RUNNER_H

#include <string>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessor.h>
#include <storm/modelchecker/multiobjective/preprocessing/SparseMultiObjectivePreprocessorResult.h>
#include <storm/storage/SparseMatrix.h>
#include <Eigen/Sparse>

namespace mopmc {

    bool run(std::string const& path_to_model, std::string const& property_string);

}

#endif //MOPMC_RUNNER_H
