#include <storm/api/storm.h>
#include <storm/utility/initialize.h>

#include "mopmc-src/ExplicitModelBuilder.h"


int main (int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Needs exactly 2 arguments: model file and property" << std::endl;
        return 1;
    }

    // Init loggers
    storm::utility::setUp();
    // Set some settings objects.
    storm::settings::initializeAll("storm-starter-project", "storm-starter-project");

    // Call function
    // Keep this for now to prototype but clean up with automatic model
    // selection later on once everything is finished.
    //auto result = mopmc::check(argv[1], argv[2]);
    mopmc::stormCheck(argv[1], argv[2]);
    // And print result
    //std::cout << "Result > 0.5? " << (result ? "yes" : "no") << std::endl;
    return 0;
}
