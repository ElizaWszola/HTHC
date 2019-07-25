# HTHC - Heterogeneous Tasks on Homogeneous Cores

## What is HTHC?

HTHC is a tool for fast training of generalized linear models (GLM) with a heterogeneous algorithm working on a single CPU. The algorithm uses block asynchronous coordinate descent with duality-gap based coordinate selection. It consists of two separate tasks working in parallel:

- Task A selects the most relevant coordinates, based on their contribution to the duality gap of the trained model, and accumulates them in a block. In a setting with high bandwidth memory, it operates on regular memory (such as DRAM).
- Task B performs updates on the block of the selected coordinates. In a setting with high bandwidth memory, it operates on standard memory (such as Xeon Phi's MCDRAM).

The currently supported models are Lasso and SVM.

The target machines are the Intel Knights Landing and Knights Mill processors.

For theory and implementation details, see

[1] Eliza Wszola, Celestine Mendler-Dünner, Martin Jaggi, Markus Püschel, [On Linear Learning with Manycore Processors](https://arxiv.org/abs/1905.00626) (the implementation)

[2] Celestine Dünner, Thomas Parnell, Martin Jaggi, [Efficient Use of Limited-Memory Accelerators for Linear Learning on Heterogeneous Systems](https://arxiv.org/abs/1708.05357) (the theory)

Additionally, HTHC uses a modified subset of the [Clover Library](https://github.com/astojanov/Clover) (the original implementation by Alen Stojanov and Tyler Michael Smith).

## Requirements

The code has been verified to compile and run under Linux CentOS.

Other requirements:

### With quantized representation ENABLED:

- [Cmake](https://cmake.org) 2.8.1 or greater.

- [Intel C++ Compiler](https://software.intel.com/en-us/c-compilers) 17.0.0.20160721 or greater.

- [Intel Math Kernel Library](https://software.intel.com/en-us/mkl) 2017.0.0 or greater.

- [Intel Integrated Performance Primitives](https://software.intel.com/en-us/intel-ipp) 2017.0.0 or greater.

- AVX-512 or AVX2 support. AVX-512 is required for good performance, but can be disabled (see [Configuration](#configuration)).

- (optional, but HIGHLY RECOMMENDED) [Memkind](https://memkind.github.io/memkind) and high bandwidth memory support are recommended good performance. Can be disabled (see [Configuration](#configuration)).

### With quantized representation DISABLED (see [Configuration](#configuration)]):

- [Cmake](https://cmake.org) 2.8.1 or greater.

- [Intel C++ Compiler](https://software.intel.com/en-us/c-compilers) 17.0.0.20160721 or greater **OR** [GCC](https://gcc.gnu.org) 4.9 or greater - if GCC is preferred, edit `CMakeLists.txt` and set `CMAKE_CXX_COMPILER` to "g++".

- AVX-512 or AVX2 support. AVX-512 is required for good performance, but can be disabled (see [Configuration](#configuration)).

- (optional, but HIGHLY RECOMMENDED) [Memkind](https://memkind.github.io/memkind) and high bandwidth memory support are recommended good performance. Can be disabled (see [Configuration](#configuration)).

## Build

### The HTHC executable

To build the HTHC executable, make sure that your system fulfills all the requirements and run

```
git clone https://github.com/ElizaWszola/HTHC.git
cd HTHC
cmake .
make
```
If some of the optional requirements are not fulfilled, the support for related functionality can be disabled (see [Configuration](#configuration)).

The resulting executable is named `hthc`.

## The parser

To build the parser to transform the data from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format to the internal format used by HTHC, enter the HTHC directory and run
```
cd data
make
```
The resulting executable is named `parse`.

## Input preprocessing

The data files must be first parsed to the representation-specific format from the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format. To do this, use the [the parser](#the-parser) provided with the implementation.
```
./parse <input_file_name> <n_samples> <n_features> <output_file_name> <primal|dual> <sparse|dense>
```
The code creates separate files for X and Y values, primal and dual representations, as well as sparse and dense data. For quantized computations, use dense data.

For a LIBSVM-formatted file `my_dataset` the following files are generated for various options:

- primal dense: `my_datasetX`, `my_datasetY`

- primal sparse: `my_dataset_sparseX`, `my_datasetY`

- dual dense: `my_dataset_dualX`, `my_datasetY`

- dual sparse: `my_dataset_dual_sparseX`, `my_datasetY`

### On data format

The data size is not stored in the data files and must be provided as arguments to the `hthc` program.

#### Y values

The Y values are stored as a binary array of floating point numbers (one number per value).

For example, i-th sample's value is accessed with `Y[i]`.

#### Dense X values

In the dense format, the X values are stored as a binary 1D array of floating point numbers (one number per feature value, zero values are stored explicitly). The array structure corresponds to a flattened 2D array.

For example, j-th feature of i-th sample is accessed with `X[i * n_features + j]` in the primal form and `X[j * n_samples + i]` in the dual form.

#### Sparse X values

In the sparse format, the X values are stored in binary format, sample-by-sample. For each sample, the following data are stored consecutively:

- NNZ elements (uint32_t)
- All non-zero indices (an array of uint32_t of length NNZ)
- All non-zero values (an array of float of length NNZ)

## Running HTHC

To run the compiled `hthc` executable, run `./hthc` with the suitable options.

### Input

The usage message is:

```
hthc <lasso|svm> <name> <samples> <features> [args]

Default arguments are defined in macros in main.cpp.
The train dataset must be provided in the arguments.
HTHC uses preprocessed data files named with the pattern:
  [name]X, [name]Y, [name]_sparseX, [name]_dualX (...)
To run for a specific dataset, use only its [name].
See README.md for details.

Arguments:
-h            : print usage
-s <name> <n> : test dataset name, test samples ["" 0]
-b <n>        : % of data on Task B [25]
-mr <n>       : maximum number of rounds (epochs) [3000]
-mt <n>       : maximum time (in s) [2e18]
-l <n>        : regularization parameter lambda [1e-4]
-e <n>        : convergence criterion epsilon [1e-5]
-ta <n>       : parallel updates on A [2]
-tb <n>       : parallel updates on B [6]
-vb <n>       : threads per vector on B [1]
--b-only      : disable A, B executes on all data [false]
-dr <0|1|2>   : data representation: [0]
    0         :     dense 32-bit
    1         :     sparse 32-bit
    2         :     dense 32/quantized 4-bit
-sl <n>       : sparse piece length (power of 2) [64]
-ll <n>       : b lock length (power of 2) [1024]
--verbose     : verbose [false]
--use-omp     : use OpenMP (baseline, dense only) [false]
```

The first four arguments correspond to the following parameters:

- Optimization model: lasso uses primal representation and SVM uses dual representation.

- Dataset name.

- Number of samples.

- Number of features.

Let us break down the other arguments:

- `-h` - prints usage.

- `-s <name> <n>` - picks a test file with n samples. The number of features is assumed to be the same as in the train file. The file should be pre-processed to the dual format. If the train file is dense, a dense test file is required. If the train file is sparse, a sparse test file is required. The default values are an empty string and 0 (no test).

- `-b <n>` - marks how many % of the original data should be processed by Task B in each epoch. It is equivalent to the coordinate descent block size. The default value is 25.

- `-mr <n>` - the maximum number of rounds (epochs) allowed for the optimization. The optimization stops when this number is exceeded. Each epoch consists of a full update of data accessed by Task B (coordinate descent block) and marks a synchronization point between the Tasks A and B, when new coordinates are selected and copied to Task B's memory. The default value is 3000.

- `-mt <n>` - the maximum time allowed for the optimization (does not include testing). The optimization stops when this time is exceeded. The default value is 2e18.

- `-l <n>` - the regularization parameter λ. This should be inverse to the regularization parameter `C` used by [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) and related libraries (`λ = 1/C`). The default value is 1e-4.

- `-e <n>` - the regularization parameter ε marking the minimal difference in the total duality gap. The optimization stops when the difference goes below ε. The default value is 1e-4.

- `-ta <n>` - the number of threads used by Task A (number of parallel updates on Task A). The default value is 2.

- `-tb <n>` - the number of parallel updates on Task B. The default value is 6.

- `-vb <n>` - the number of threads per each parallel update of Task B. These threads are used to parallelize algebraic computations per-vector. The total number of threads used by Task B is therefore `tb * vb`. Note that for the data in sparse format, `vb` should be always set to 1. The default value is 1.

- `--b-only` - disables the heterogeneous scheme. No blocks are selected and in each epoch, Task B performs an update on the entire dataset. This mode is akin to cyclic asynchronous coordinate descent and is used as a baseline. In this setting, `ta` should still be a non-zero value, as Task A still calculates the duality gap outside the epochs to measure the quality of the obtained model. The updates of Task A are not timed. The default value is false.

- `-dr <0|1|2>` - the enum for data representation. Can be either 32-bit dense (0), 32-bit sparse (1), or a combination of dense quantized 4-bit for the dataset and dense 32-bit for anything else (2). The default is value is 0 (dense).

- `-sl <n>` - The length of sparse piece (used for sparse representations for efficient copies between A and B - does not apply to the B-only setting). Should be a power of 2 and depends on data density. Ideally, should be close to the average number of the nonzero numbers in the dataset. The default value is 64.

- `-ll <n>` - The length of the lock used for the updates of optimization parameters. The default value is 1024 and works well for dense data. For sparse data, depends on the density. The longer and sparser the feature vectors, the greater numbers are preferred.

- `--verbose` - Enable coarse information on the progress of the program. The default value is false.

- `--use-omp` - Use a simple OpenMP implementation. Works only for dense data and should be used only as a baseline. The default value is false.

Note that the total number of threads used by the optimization (`ta + tb * vb`) should not exceed the number of logical threads on the target machine.

### Output

For the heterogeneous setting, the HTHC code outputs the following values in the CSV format:

- `round` - current epoch.

- `cost` - current objective function value.

- `duality_gap` - current duality gap, as seen by Task A. The real gap might be slightly different as this column uses information provided by A's updates.

- `#z_i_updates` - number of gap updates performed in the current epoch.

- `#swaps` - number of coordinates copied between A and B in the current epoch. If a coordinate is already on B, it will not be copied, so this number should decrease over time as the most relevant coordinates are identified by the scheme.

- `t_swap` - time required to perform the coordinate copy (in ns).

- `t_compute` - time required to perform the update on B (in ns).

- `t_find_set` - time required to identify the new most relevant coordinates (in ns).

- `t_tot` - total time so far, measured as the sum of the above.

Additionally, if the user chooses a test dataset, the following information is printed:

- `accuracy` - classification accuracy.

- `precision` - classification precision.

- `recall` - classification recall.

- `f1` - classification F1 score.

- `mean_squared` - regression mean squared error.

Use the relevant values, depending on whether you intend to perform classification or regression.

For the "B-only" setting, the irrelevant values are not printed.

At the end of the optimization, a row of values is printed. They stand for (in the given order):

B size (%), `ta`, `tb`, `vb`, total epochs passed, objective function value, duality gap, average updates of Task A per epoch, average `t_swap`, average `t_compute`, average `t_update`, total optimization time.

## Configuration

In order to trouble shoot problems with support for specific functionality or play around with the configuration, a few variables are provided and highlighted in the `CMakeLists.txt` file.

#### DISABLE_AVX512
When set to true, the HTHC drops explicit AVX-512 intrinsics, enables an AVX2 flag, and runs on scalar code, letting the compiler decide how to vectorize. The Clover vectors still rely on explicit AVX2 intrinsics.

#### DISABLE_HBW
If no high bandwidth memory is available on the target machine, the use of HBW can be disabled with this flag. Tasks A and B will then both use the same type of memory (e.g. DRAM), but still allocate separate blocks for their own computations and periodically copy data between one another.

#### DISABLE_LOCKS
This variable can be used to trade accuracy for speed. When set, the updates on optimization parameters are lock-free, in a ["HOGWILD!"](https://arxiv.org/abs/1106.5730) fashion. This can improve optimization speed (especially for sparse data), but invalidates some theoretical guarantees on convergence.

#### DISABLE_QUANTIZED
The quantized representation depends on an underlying library which has a few dependencies and can be compiled only with the [Intel C++ Compiler](https://software.intel.com/en-us/c-compilers). As this representation is not an essential part of the code, it is safe to disable it and work with the 32-bit formats only.

## License

This code is open source and licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
