/**
 * Copyright 2019 Eliza Wszola (eliza.wszola@inf.ethz.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "task_a.h"
#include "task_b.h"
#include "measurements.h"
#include "threaded.h"
#include "readers.h"

//defaults
#define FILE_NAME ""
#define TEST_FILE_NAME ""
#define SAMPLES 0
#define FEATURES 0
#define TEST_SAMPLES 0
#define B_SIZE 25
#define ROUNDS 3000
#define LAMBDA 1e-4
#define EPSILON 1e-5
#define A_UPDATES 2
#define B_UPDATES 6
#define B_THREADS_PER_VEC 1
#define MAX_TIME 2e18

#define DEFAULT_LINE_SIZE 64
#define DEFAULT_SPARSE_PIECE_LENGTH 64
#define DEFAULT_B_THREAD_CHUNK_SIZE 1024
#define DEFAULT_REP DENSE32

std::string file_name = FILE_NAME;
std::string test_file_name = TEST_FILE_NAME;
uint64_t samples = SAMPLES;
uint64_t features = FEATURES;
uint64_t test_samples = TEST_SAMPLES;
real b_size = B_SIZE;
uint64_t rounds = ROUNDS;
uint64_t a_par_updates = A_UPDATES;
uint64_t b_par_updates = B_UPDATES;
uint64_t b_threads_per_vec = B_THREADS_PER_VEC;
uint64_t max_time = MAX_TIME;
real lambda = LAMBDA;
real epsilon = EPSILON;
bool primal = true;
bool b_only = false;
bool use_omp = false;
bool verbose = false;

void print_usage() {
  std::cout << "====USAGE====\n"
      << "hthc <lasso|svm> <name> <samples> <features> [args]\n\n"
      << "Default arguments are defined in macros in main.cpp.\n"
      << "The train dataset must be provided in the arguments.\n"
      << "HTHC uses preprocessed data files named with the pattern:\n"
      << "  [name]X, [name]Y, [name]_sparseX, [name]_dualX (...)\n"
      << "To run for a specific dataset, use only its [name].\n"
      << "See README.md for details.\n\n" 
      << "Arguments:\n"
      << "-h            : print usage\n"
      << "-s <name> <n> : test dataset name, test samples [\"\" 0]\n"
      << "-b <n>        : % of data on Task B [25]\n"
      << "-mr <n>       : maximum number of rounds (epochs) [3000]\n"
      << "-mt <n>       : maximum time (in s) [2e18]\n"
      << "-l <n>        : regularization parameter lambda [1e-4]\n"
      << "-e <n>        : convergence criterion epsilon [1e-5]\n"
      << "-ta <n>       : parallel updates on A [2]\n"
      << "-tb <n>       : parallel updates on B [6]\n"
      << "-vb <n>       : threads per vector on B [1]\n"
      << "--b-only      : disable A, B executes on all data [false]\n"
      << "-dr <0|1|2>   : data representation: [0]\n"
      << "    0         :     dense 32-bit\n"
      << "    1         :     sparse 32-bit\n"
      #if HAS_QUANTIZED
      << "    2         :     dense 32/quantized 4-bit\n"
      #endif
      << "-sl <n>       : sparse piece length (power of 2) [64]\n"
      << "-ll <n>       : b lock length (power of 2) [1024]\n"
      << "--verbose     : verbose [false]\n"
      << "--use-omp     : use OpenMP (baseline, dense only) [false]\n";

  exit(0);
}

void args_error() {
  std::cout << "Invalid argument format or values.\n";
  print_usage();
}

void validate_args() {
  uint32_t cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (samples <= 0 || features <= 0 || test_samples < 0 || rounds <= 0
      || epsilon <= 0)
    args_error();
  if (b_only){
    if (b_par_updates <= 0 || b_threads_per_vec <= 0
        || b_par_updates * b_threads_per_vec > cores)
      args_error();
  } else {
    if (a_par_updates <= 0 || b_par_updates <= 0
        || b_threads_per_vec <= 0
        || a_par_updates + b_par_updates * b_threads_per_vec > cores)
      args_error();
  }
  if (!b_only && b_size <= 0)
    args_error();
  if (use_omp && data_rep != DENSE32)
    args_error();
}

void read_args(int argc, char *argv[]) {
  if (argc <= 1 || !std::strcmp(argv[1], "-h") || argc < 5) {
    print_usage();
  } else {
    if (!std::strcmp(argv[1], "lasso"))
      primal = true;
    else if (!std::strcmp(argv[1], "svm"))
      primal = false;
    else
      args_error();
    if (argc < 5)
      print_usage();
    file_name = argv[2];
    samples = std::atoi(argv[3]);
    features = std::atoi(argv[4]);
    for (uint32_t i = 5; i < argc; ++i) {
      if (!std::strcmp(argv[i], "-d")) {
        if (i + 3 < argc) {
          file_name = argv[i + 1];
          samples = std::atoi(argv[i + 2]);
          features = std::atoi(argv[i + 3]);
          if (samples <= 0 || features <= 0)
            args_error();
          i += 3;
        } else {
          args_error();
        }
      } else if (!std::strcmp(argv[i], "-s")) {
        if (i + 2 < argc) {
          test_file_name = argv[i + 1];
          test_samples = std::atoi(argv[i + 2]);
          if (samples <= 0 || features <= 0)
            args_error();
          i += 2;
        } else {
          args_error();
        }
      } else if (!std::strcmp(argv[i], "-b")) {
        ++i;
        if (i < argc)
          b_size = std::atof(argv[i]);
        else
          args_error();  
      } else if (!std::strcmp(argv[i], "-mr")) {
        ++i;
        if (i < argc)
          rounds = std::atoi(argv[i]);
        else
          args_error();  
      } else if (!std::strcmp(argv[i], "-mt")) {
        ++i;
        if (i < argc)
          max_time = std::atoi(argv[i]) * (uint64_t)1000000000;
        else
          args_error();
      } else if (!std::strcmp(argv[i], "-l")) {
        ++i;
        if (i < argc)
          lambda = std::atof(argv[i]);
        else
          args_error();  
      } else if (!std::strcmp(argv[i], "-e")) {
        ++i;
        if (i < argc)
          epsilon = std::atof(argv[i]);
        else
          args_error();  
      } else if (!std::strcmp(argv[i], "-ta")) {
        ++i;
        if (i < argc)
          a_par_updates = std::atoi(argv[i]);
        else
          args_error();  
      } else if (!std::strcmp(argv[i], "-tb")) {
        ++i;
        if (i < argc)
          b_par_updates = std::atoi(argv[i]);
        else
          args_error();  
      } else if (!std::strcmp(argv[i], "-vb")) {
        ++i;
        if (i < argc)
          b_threads_per_vec = std::atoi(argv[i]);
        else
          args_error();
      } else if (!std::strcmp(argv[i], "-sl")) {
        ++i;
        if (i < argc) {
          uint32_t len = std::atoi(argv[i]);
          if (len > 0 && (len & (len - 1)) == 0)
            SPARSE_PIECE_LENGTH = len;
          else
            args_error();
        } else {
          args_error();
        }
      } else if (!std::strcmp(argv[i], "-ll")) {
        ++i;
        if (i < argc) {
          uint32_t len = std::atoi(argv[i]);
          if (len > 0 && (len & (len - 1)) == 0)
            B_THREAD_CHUNK_SIZE = len;
          else
            args_error();
        } else {
          args_error();
        }
      } else if (!std::strcmp(argv[i], "-dr")) {
        ++i;
        uint32_t rep;
        if (i < argc)
          rep = std::atoi(argv[i]);
        else
          args_error();
        if (rep == 0)
          data_rep = DENSE32;
        else if (rep == 1)
          data_rep = SPARSE32;
        #if HAS_QUANTIZED
        else if (rep == 2)
          data_rep = QUANTIZED;
        #endif
        else
          args_error();
      } else if (!std::strcmp(argv[i], "--b-only")) {
        b_only = true;
      } else if (!std::strcmp(argv[i], "--use-omp")) {
        use_omp = true;
      } else if (!std::strcmp(argv[i], "--verbose")) {
        verbose = true;
      } else {
        std::cout << "Unknown parameter " << argv[i] << ", ignoring\n";
      }
    }
    if (file_name.length() == 0)
      print_usage();
    validate_args();
  }
}

struct MainArgs {
  TaskB *task_b;
};

void *run_lasso_b(void *data) {
  MainArgs *args = (MainArgs*)data;
  args->task_b->run_lasso();
}

void *run_svm_b(void *data) {
  MainArgs *args = (MainArgs*)data;
  args->task_b->run_svm();
}

int main(int argc, char *argv[]) {
  
  SPARSE_PIECE_LENGTH = DEFAULT_SPARSE_PIECE_LENGTH;
  B_THREAD_CHUNK_SIZE = DEFAULT_B_THREAD_CHUNK_SIZE;
  data_rep = DEFAULT_REP;
  
  read_args(argc, argv);
  
  uint64_t time_b = 0;
  uint64_t time_u = 0;
  uint64_t time_p = 0;
  uint64_t time_total = 0;
  uint64_t r = 0;
  
  bool x_hbw = false;
  bool y_hbw = b_only ? B_USE_HBW : A_USE_HBW;
  clocktime start_b, end_b;
  clocktime start_u, end_u;
  clocktime start_p, end_p;
  OneSparseMatrix x_sparse;
  QuantMatrix x_quant;
  Matrix x_dense;
  Vector y;
  Vector weights;
  double objective, gap;
  uint64_t total_updated;
  bool a_running = true;
  uint32_t swaps = 0;
  uint64_t elements = primal ? features : samples;
  uint64_t b_elements = std::ceil(b_size / 100.0 * elements);
  pthread_t b_thread;
  double positives, negatives, true_positives, true_negatives;
  double false_positives, false_negatives, ground_y, obtained_y;
  double accuracy, precision, recall, f1, mean_squared;
  uint64_t total_a_up;
  
  if (use_omp)
    omp_set_nested(true);
  LINE_SIZE = ((data_rep == QUANTIZED) ? 128 : 64);

  OneSparseMatrix x_t_sparse;
  Matrix x_t_dense;
  Vector y_t;
  
  if (verbose)
    std::cout << "Loading data...\n";

  if (data_rep == DENSE32) {
    if (primal)
      x_dense = read_matrix_from_binary_primal(file_name, samples,
          features, x_hbw);
    else
      x_dense = read_matrix_from_binary_dual(file_name, samples,
          features, x_hbw);
    if (test_samples > 0)
      x_t_dense = read_matrix_from_binary_dual(test_file_name,
          test_samples, features, false);
  } else if (data_rep == SPARSE32) {
    if (primal)
      x_sparse = read_sparse_matrix_from_binary_primal(file_name,
          samples, features, x_hbw);
    else
      x_sparse = read_sparse_matrix_from_binary_dual(file_name,
          samples, features, x_hbw);
    if (test_samples > 0)
      x_t_sparse = read_sparse_matrix_from_binary_dual(test_file_name,
          test_samples, features, false);
  } else if (data_rep == QUANTIZED) {
    if (primal)
      x_quant = read_quantized_matrix_from_binary_primal(file_name,
          samples, features, x_hbw);
    else
      x_quant = read_quantized_matrix_from_binary_dual(file_name,
          samples, features, x_hbw);
    if (test_samples > 0)
      x_t_dense = read_matrix_from_binary_dual(test_file_name,
          test_samples, features, false);
  }
  y = read_vector_from_binary(file_name, samples, y_hbw);
  if (test_samples > 0)
    y_t = read_vector_from_binary(test_file_name, test_samples, false);

  if (verbose)
    std::cout << "Loaded.\n";
  
  uint32_t pieces = 0;
  uint32_t* sparse_weights;
  if (data_rep == SPARSE32) {
    b_elements = std::ceil(b_size / 100.0 * elements);
    sparse_weights = (uint32_t*)b_malloc(elements * sizeof(uint32_t),
        false);
    for (uint32_t i = 0; i < elements; ++i)
      sparse_weights[i] = std::ceil(get_column(x_sparse, i).padded_nnz
          / (double)SPARSE_PIECE_LENGTH);
    std::sort(sparse_weights, sparse_weights + elements);
    for (uint32_t i = elements - b_elements; i < elements; ++i)
      pieces += sparse_weights[i];
  } else {
    b_elements = std::ceil(b_size / 100.0 * elements);
  }
  
  TaskA *task_a = (TaskA*)b_malloc(sizeof(TaskA), A_USE_HBW);
  TaskB *task_b = (TaskB*)b_malloc(sizeof(TaskB), B_USE_HBW);
  if (b_only) {
    if (verbose)
      std::cout << "Initializing B...\n";
    if (data_rep == DENSE32) {
      task_b->initialize_all(x_dense, y, samples, features, lambda,
          b_par_updates, b_threads_per_vec, &a_running, primal);
      task_a->initialize(x_dense, y, samples, features, elements,
          lambda, a_par_updates, b_par_updates, b_threads_per_vec,
          &a_running, primal, b_only);
    } else if (data_rep == SPARSE32) {
      task_b->initialize_all(x_sparse, y, samples, features, lambda,
          b_par_updates, b_threads_per_vec, &a_running, primal);
      task_a->initialize(x_sparse, y, samples, features, elements,
          lambda, a_par_updates, b_par_updates, b_threads_per_vec,
          &a_running, primal, b_only);
    } else if (data_rep == QUANTIZED) {
      task_b->initialize_all(x_quant, y, samples, features, lambda,
          b_par_updates, b_threads_per_vec, &a_running, primal);
      task_a->initialize(x_quant, y, samples, features, elements,
          lambda, a_par_updates, b_par_updates, b_threads_per_vec,
          &a_running, primal, b_only);
    }
  } else {
    if (verbose)
      std::cout << "Initializing A...\n";
    if (data_rep == DENSE32) {
      task_a->initialize(x_dense, y, samples, features, b_elements,
          lambda, a_par_updates, b_par_updates, b_threads_per_vec,
          &a_running, primal, b_only);
    } else if (data_rep == SPARSE32) {
      task_a->initialize(x_sparse, y, samples, features, b_elements,
          lambda, a_par_updates, b_par_updates, b_threads_per_vec,
          &a_running, primal, b_only);
    } else if (data_rep == QUANTIZED) {
      task_a->initialize(x_quant, y, samples, features, b_elements,
          lambda, a_par_updates, b_par_updates, b_threads_per_vec,
          &a_running, primal, b_only);
    }
    if (verbose)
      std::cout << "Initializing B...\n";
    task_b->initialize(samples, features, pieces, b_elements, lambda,
        b_par_updates, b_threads_per_vec, &a_running, primal);
  }
  if (data_rep == SPARSE32)
    b_free(sparse_weights, false);

  if (verbose)
    std::cout << "Learning..." << std::endl;
  
  if (b_only) {
    std::cout << "round,cost,duality_gap,t_compute,t_tot";
  } else {
    std::cout << "round,cost,duality_gap,#z_i_updates,#swaps,t_swap,"
        << "t_compute,t_find_set,t_tot";
  }
  if (test_samples > 0)
    std::cout << ",accuracy,precision,recall,f1,mean_squared";
  std::cout << std::endl;
  
  MainArgs args = {task_b};

  gap = 0;
  r = 0;
  total_a_up = 0;
  time_b = 0;
  time_p = 0;
  time_u = 0;
  time_total = 0;

  while (((gap >= epsilon && r < rounds) || r == 0)
      && time_total < max_time) {
    ++r;
    a_running = true;
    if (!b_only) {
      get_time(&start_u);
      if (data_rep == DENSE32)
        task_b->update(task_a->a_p, task_a->a_a, task_a->a_b,
            task_a->a_alpha, task_a->a_norms, primal, swaps);
      else if (data_rep == SPARSE32)
        task_b->update(task_a->a_p, task_a->a_a_one_sparse, task_a->a_b,
            task_a->a_alpha, task_a->a_norms, primal, swaps);
      else if (data_rep == QUANTIZED)
        task_b->update(task_a->a_p, task_a->a_a_quant, task_a->a_b,
            task_a->a_alpha, task_a->a_norms, primal, swaps);
      get_time(&end_u);
      time_u += get_time_difference(&start_u, &end_u);
    }
    get_time(&start_b);
    if (use_omp) {
      if (primal) {
        #pragma omp parallel num_threads(2)
        {
          if (omp_get_thread_num() == 0) {
            task_b->run_lasso_omp();
          } else if (!b_only) {
            task_a->shuffle(primal);
            task_a->run_lasso_omp();
          }
        }
      } else {
        #pragma omp parallel num_threads(2)
        {
          if (omp_get_thread_num() == 0) {
            task_b->run_svm_omp();
          } else if (!b_only) {
            task_a->shuffle(primal);
            task_a->run_svm_omp();
          }
        }
      }
    } else {
      if (primal) {
        pthread_create(&b_thread, NULL, run_lasso_b, (void*)&args);
        if (!b_only)
          task_a->run_lasso();
      } else {
        pthread_create(&b_thread, NULL, run_svm_b, (void*)&args);
        if (!b_only)
          task_a->run_svm();
      }
      pthread_join(b_thread, NULL);
    }
    get_time(&end_b);
    time_b += get_time_difference(&start_b, &end_b);
    if (b_only) {
      task_a->update_parameters(task_b->b_alpha, task_b->b_w);
    } else {
      get_time(&start_p);
      task_a->update_alpha(task_b->b_alpha, task_b->b_p,
          task_b->get_b_size());
      task_a->update_w(task_b->b_w);
      get_time(&end_p);
      time_p += get_time_difference(&start_p, &end_p);
    }
    if (b_only)
      time_total = time_b;
    else
      time_total = time_b + time_p + time_u;
    if (b_only) {
      a_running = true;
      if (primal) {
        task_a->run_lasso();
        task_a->update_lasso_stats();
      } else {
        task_a->run_svm();
        task_a->update_svm_stats();
      }
      a_running = false;
    } else {
      if (primal)
        task_a->update_lasso_stats();
      else
        task_a->update_svm_stats();
      get_time(&start_p);
        task_a->update_p(primal);
      get_time(&end_p);
      time_p += get_time_difference(&start_p, &end_p);
    }
    objective = task_a->get_objective();
    gap = task_a->get_duality_gap();
    if (b_only) {
      std::cout << r << "," << objective << "," << gap << ","
          << get_time_difference(&start_b, &end_b) << ","
          << time_total;
    } else {
      total_updated = task_a->get_total_updated();
      total_a_up += total_updated;
      std::cout << r << "," << objective << "," << gap << ","
          << total_updated << "," << swaps << ","
          << get_time_difference(&start_u, &end_u) << ","
          << get_time_difference(&start_b, &end_b) << ","
          << get_time_difference(&start_p, &end_p) << ","
          << time_total;
    }
    if (test_samples > 0) {
      Vector weights = task_a->get_weights(primal);
      if (data_rep == QUANTIZED && !primal)
        inner_transpose(weights);
      positives = 0;
      negatives = 0;
      true_positives = 0;
      true_negatives = 0;
      false_positives = 0;
      false_negatives = 0;
      mean_squared = 0;
      for (uint64_t i=0; i<test_samples; ++i) {
        ground_y = get_value(y_t, i);
        obtained_y = 0;
        #if SCALAR
        if (data_rep == DENSE32)
          obtained_y = dot_product(weights, get_column(x_t_dense, i));
        else if (data_rep == SPARSE32)
          obtained_y = dot_product(weights,
              get_column(x_t_sparse, i));
        else if (data_rep == QUANTIZED)
          obtained_y = dot_product(weights, get_column(x_t_dense, i));
        #else
        if (data_rep == DENSE32)
          obtained_y = dot_product_v(weights, get_column(x_t_dense, i));
        else if (data_rep == SPARSE32)
          obtained_y = dot_product_v(weights,
              get_column(x_t_sparse, i));
        else if (data_rep == QUANTIZED)
          obtained_y = dot_product_v(weights, get_column(x_t_dense, i));
        #endif
        mean_squared += (ground_y - obtained_y)
          * (ground_y - obtained_y);
        if (ground_y > 0) {
              positives++;
              if (obtained_y > 0)
          true_positives++;
              else
          false_negatives++;
        } else if (ground_y < 0) {
              negatives++;
              if (obtained_y < 0)
          true_negatives++;
              else
          false_positives++;
        }
      }
      if (data_rep == QUANTIZED && !primal)
        inner_transpose(weights);
      mean_squared /= test_samples;
      accuracy = (true_positives + true_negatives) * 100.0
	      / test_samples;
      precision = true_positives / (true_positives + false_positives);
      recall = true_positives / (true_positives + false_negatives);
      f1 = (200.0 * precision * recall) / (precision + recall);
      std::cout << "," << accuracy << "," << precision * 100 << ","
	      << recall * 100 << "," << f1 << "," << mean_squared;
    }
    std::cout << std::endl;
  }
  std::cout << b_size << "," << a_par_updates << "," << b_par_updates
      << "," << b_threads_per_vec << "," << r << "," << objective
      << "," << gap << "," << total_a_up / r << "," << time_u / r
      << "," << time_b / r << "," << time_p / r << "," << time_total
      << std::endl;

  if (b_only) {
    if (verbose)
      std::cout << "Deinitializing B...\n";
    task_b->deinitialize();
    task_a->deinitialize();
  } else {
    if (verbose)
      std::cout << "Deinitializing A...\n";
    task_a->deinitialize();
    if (verbose)
      std::cout << "Deinitializing B...\n";
    task_b->deinitialize();
    if (verbose)
      std::cout << "Deinitializing AB...\n";
    task_b->deinitialize_ab();
  }
  
  b_free(task_a, A_USE_HBW);
  b_free(task_b, B_USE_HBW);

  if (verbose)
    std::cout << "Deinitializing data...\n";
  if (data_rep == DENSE32)
    destroy (x_dense, x_hbw);
  else if (data_rep == SPARSE32)
    destroy (x_sparse, x_hbw);
  else if (data_rep == QUANTIZED)
    destroy (x_quant, x_hbw);
  destroy (y, y_hbw);
  if (test_samples > 0) {
    if (data_rep == SPARSE32)
      destroy (x_t_sparse, false);
    else
      destroy (x_t_dense, false);
    destroy (y_t, false);
  }
    
  if (verbose)
    std::cout << "Done.\n";
  return 0;
  
}
