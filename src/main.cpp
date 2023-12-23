
#include <iostream>
#include <boost/program_options.hpp>
#include "mopmc-src/Runner.h"
#include "mopmc-src/QueryOptions.h"

namespace po = boost::program_options;
using namespace std;

int main (int ac, char *av[]) {

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("prism", po::value<string>(), "prism model file")
                ("props", po::value<string>(), "property file")
                ("fn", po::value<string>(), "convex function")
                ("popt", po::value<string>(), "primary optimizer")
                ;
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        string modelFile, propsFile;
        if (vm.count("prism") && vm.count("props")) {
            modelFile = vm["prism"].as<string>();
            propsFile = vm["props"].as<string>();
        }

        mopmc::QueryOptions queryOptions{};
        if (vm.count("fn")) {
            const auto& s = vm["fn"].as<string>();
            if (s == "mse") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::MSE;
            } else if (s == "se") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::SE;
            } else if (s == "var") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::VAR;
            } else if (s == "sd") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::SD;
            } else {
                cout << "not supported convex function\n";
                return 1;
            }
        }
        if (vm.count("popt")) {
            const auto& s = vm["popt"].as<string>();
            if (s == "away-step") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::AWAY_STEP;
            } else if (s == "linopt") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::LINOPT;
            } else if (s == "blended") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::BLENDED;
            } else if (s == "blended-step-opt") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::BLENDED_STEP_OPT;
            } else if (s == "pgd") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::PGD;
            }
            else {
                cout << "not supported primary optimizer\n";
                return 1;
            }
        }
        mopmc::run(modelFile, propsFile, queryOptions);
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
        return 1;
    }
    return 0;
}
