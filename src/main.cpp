#include <storm/api/storm.h>
#include <storm/utility/initialize.h>
#include <filesystem>
//#include <storm/settings/

#include "mopmc-src/ExplicitModelBuilder.h"
#include "mopmc-src/Runner.h"

int main (int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Needs exactly 2 arguments: model file and property" << std::endl;
        return 1;
    }

    // Init loggers
    storm::utility::setUp();
    // Set some settings objects.
    storm::settings::initializeAll("storm-starter-project", "storm-starter-project");

    // Accepts Prism file format
    //mopmc::check(argv[1], argv[2]);
    mopmc::run(argv[1], argv[2]);
    return 0;
}
