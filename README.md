# MOPMC with LTL Property Specifications
Multi-objective Probabilistic Model Checking built on a C++ API of [Storm](https://www.stormchecker.org).
This project is built on the Storm project and to use it, Storm model checker needs to be build from 
**source** with all dependencies. See [Storm](https://www.stormchecker.org) for installation details.

## Dependencies

This build is known to work on Ubuntu 20.04 LTS. Other linux flavours are possible however dependency setup
can be tricky.

Before starting, make sure that Storm and **all of its dependencies are installed** is installed. If not, see the [documentation](https://www.stormchecker.org/documentation/obtain-storm/build.html).

This project uses cmake which should be bundled with Ninja. If Ninja is available you will be able
to make use of the convenient configurations and build script.

This project requires CUDA Toolkit 12.xx and the associated NVIDIA driver 525+. 
This cuda toolkit is essential as it provides 64bit numeric types for the GPU and provides more modern
sparse matrix multiplication algorithms from Nvidia CuSparse. If installed correctly, using the command `nvidia-smi`
you should see something like:

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.113.01             Driver Version: 535.113.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|

```

A further note on setting up your environment. 

Cuda Toolkit has a mandatory action of adding the toolkit to the `PATH` variable. Add the 
following to `.bashrc` or `.profile`:
```bash
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
```

Additionally, if you develop in an IDE which builds based off cmake, such as CLion, then you will also 
need to set the LD_LIBRARY_PATH to contain the toolkit's lib64 directory. This can also be added 
to `.bashrc` or `.profile`. 
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64 ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

This avoids errors by the IDE debug compiler relating to setting `CMAKE_CUDA_ARCHITECTURES`.

## Getting Started

First, clone and `cd` into the project then configure and compile the project. Execute
```
mkdir build
./configure.sh
./build.sh
```

To test your build is working, run the executable using the convenience script: 
```bash
./run.sh
```

This project only computes multi-objective model checking of convex queries.

## Development

`src/main.cpp` is the entry point of the project. 

The first call is to `mopmc::check` which parses a model as a Prism model along with 
properties from a `.pctl` file. These are argument inputs with the first being model and the
second being property inputs. 

Model parsing is done using Storm parsing methods and once done multi-objective model
checking is done by calling:
```c++
mopmc::multiobjective::performMultiObjectiveModelChecking(env, *mdp, formulas[0]->asMultiObjectiveFormula());
```

This class method first preprocesses the multi-objective formulas and model by calling 
methods in 
```c++
src/mopmc-src/model-checking/MultiObjectivePreprocessor.cpp(h)
```

After model construction is complete, MOPMC model checking is conducted using
the methods and classes in `src/mopmc-src/model-checking/MOPMCModelChecking.cpp(h)`.
The class often makes reference to the solvers both `c++` and `CUDA` based located in
`src/mopmc-src/solvers`.




