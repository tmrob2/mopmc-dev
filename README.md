# MOPMC with LTL Property Specifications
Multi-objective Probabilistic Model Checking built on a C++ API of [Storm](https://www.stormchecker.org).
This project is built on the Storm project and to use it, Storm model checker needs to be build from 
**source** with all dependencies. See [Storm](https://www.stormchecker.org) for installation details.

## Getting Started
Before starting, make sure that Storm is installed. If not, see the [documentation](https://www.stormchecker.org/documentation/obtain-storm/build.html).

This project uses cmake which should be bundled with Ninja. If Ninja is available you will be able 
to make use of the convenient configurations and build script.

First, clone and `cd` into the project then configure and compile the project. Execute
```
mkdir build
./configure.sh
./build.sh
```

Then, run the executable using 
```
./build/mopmc examples/multiobj_scheduler05.nm examples/multiobj_scheduler05.pctl
```

This project only computes multi-objective model checking of convex queries.

## Development

`src/main.cpp` is the entry point of the project. 

The first call is to `mopmc::stormCheck` which parses a model as a Prism model along with 
properties from a `.pctl` file. These are argument inputs with the first being model and the
second being property inputs. 

After constructing the model call `mopmc::stormtest::performMultiObjectiveModelChecking` which
initiates model checking for the model and properties input. This function first preprocesses
constructs a `class` `SparseMultiObjectivePreprocessor` and various member functions located in
`src/mopmc-src/model-checking/MultiObjectivePreprocessor.cpp(h)`

After preprocessing a new multi-objective model-checking class is constructed which 
is used to solve the problem. This is called `mopmc::multiobjective::StandardMdpPcaaChecker`
located in `src/mopmc-src/model-checking/StandardMdpPcaaChecker.cpp(h)`. This class sets up 
the model checker and does any further processing of the model according to 
weight vectors. 

Algorithm 1 is called in this class also using `StandardMdpPcaaChecker<SparseModelType>::multiObjectiveSolver`.
This function makes a call to a header `src/mopmc-src/solvers/ConvexQuery.cpp(h)` which contains
the auxiliary functions.  




