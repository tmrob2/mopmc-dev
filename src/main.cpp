#include <storm/api/storm.h>
#include <storm/utility/initialize.h>
//#include <filesystem>
//#include <storm/settings/

//#include "mopmc-src/ExplicitModelBuilder.h"
#include "mopmc-src/Runner.h"
#include "mopmc-src/QueryOptions.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
using namespace std;

int main (int ac, char *av[]) {

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("prism", po::value<string>(), "prism model file")
                ("props", po::value<string>(), "property file")
                ("fn", po::value<string>(), "convex function")
                ("popt", po::value<string>(), "primary optimizer]")
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
            const string& s = vm["fn"].as<string>();
            if (s == "mse") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::MSE;
            } else if (s == "eud") {
                queryOptions.CONVEX_FUN = mopmc::QueryOptions::EUCLIDEAN;
            } else {
                cout << "not supported convex function\n";
                return 1;
            }
        }
        if (vm.count("popt")) {
            const string& s = vm["fn"].as<string>();
            if (s == "away_step") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::AWAY_STEP;
            } else if (s == "linopt") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::LINOPT;
            } else if (s == "blended") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::BLENDED;
            } else if (s == "blended_step_opt") {
                queryOptions.PRIMARY_OPTIMIZER = mopmc::QueryOptions::BLENDED_STEP_OPT;
            }
            else {
                cout << "not supported primary optimizer\n";
                return 1;
            }
        }

        // Init loggers
        storm::utility::setUp();
        storm::settings::initializeAll("storm-starter-project", "storm-starter-project");

        mopmc::run(modelFile, propsFile, queryOptions);
        //mopmc::run(modelFile, propsFile);
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    return 0;

    if (ac < 3) {
        std::cout << "Needs exactly 2 arguments: model file and property" << std::endl;
        return 1;
    }

    // Init loggers
    storm::utility::setUp();
    // Set some settings objects.
    storm::settings::initializeAll("storm-starter-project", "storm-starter-project");

    // Accepts Prism file format
    //mopmc::check(argv[1], argv[2]);
    //mopmc::run(argv[1], argv[2]);
    return 0;
}
