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

#ifndef THREADED_H
#define THREADED_H

#include <unistd.h>
#include <ctime>
#include "omp.h"
#include "algebra.h"
#include "vectorized.h"

#define CORE_OFFSET 0
#define TILE_MULTIPLIER 1 // run B on every ith core
#define SLEEP_TIME 0

namespace threaded {
  
  struct SolverArguments {
    real regularization;
    SparseMatrix b_a_sparse;
    OneSparseMatrix b_a_one_sparse;
    QuantMatrix b_a_quant;
    Matrix b_a;
    Vector b_w;
    #if HAS_QUANTIZED
    CloverVector32* b_acol32;
    #endif
    Vector b_alpha;
    Vector b_b;
    Vector b_a_col;
    Vector b_norms;
    uint32_t data_len;
    uint32_t b_size;
    uint32_t thread_size;
    uint32_t par_updates;
    uint32_t threads_per_vec;
    uint32_t *exe_order;
    real *prods;
    real *alpha_diffs;
    
    pthread_mutex_t* cond_mutex;
    pthread_cond_t* cond;
    pthread_mutex_t* bar_mutex;
    uint32_t* bar1;
    uint32_t* bar2;
    bool* can_start;
    bool* threads_running;
    bool primal;
    bool no_chunk;
  };

  struct SolverThreadArguments {
    uint32_t vec_thread_id;
    uint32_t update_thread_id;
    struct SolverArguments *args;
  };
    
  struct GapArguments {
    real regularization;
    real ln;
    OneSparseMatrix a_a_one_sparse;
    QuantMatrix a_a_quant;
    Matrix a_a;
    Vector a_w;
    #if HAS_QUANTIZED
    CloverVector32* a_acol32;
    #endif
    Vector a_realw;
    Vector a_realz;
    Vector a_alpha;
    Vector a_z;
    Vector a_b;
    Vector a_norms;
    real bound;
    uint32_t n_gaps;
    uint32_t par_updates;
    uint32_t *exe_order;
    bool *running;
    uint32_t *updated;
    pthread_mutex_t* cond_mutex;
    pthread_cond_t* cond;
    pthread_mutex_t* bar_mutex;
    uint32_t* bar1;
    uint32_t* bar2;
    bool* can_start;
    bool* threads_running;
    bool primal;
    bool b_only;
    std::minstd_rand0* random_engine;
  };

  struct GapThreadArguments {
    uint32_t update_thread_id;
    struct GapArguments *args;
  };
  
  void initialize_b(uint32_t par_updates, uint32_t threads_per_vec,
      uint32_t atoms, uint32_t data_len, uint32_t p, bool use_hbw);
  void deinitialize_b(uint32_t par_updates, uint32_t threads_per_vec,
      uint32_t atoms, bool use_hbw);
  void initialize_a(uint32_t par_updates, uint32_t columns,
      bool use_hbw);
  void deinitialize_a(uint32_t par_updates, bool use_hbw);
  
  void reset_watoms();
  void reset_pctr();

  void *update_z_lasso_dense(void *data);
  void *update_z_svm_dense(void *data);    
  void *run_lasso_dense(void *data);
  void *run_svm_dense(void *data);
  void *update_z_lasso_sparse(void *data);
  void *update_z_svm_sparse(void *data);    
  void *run_lasso_sparse(void *data);
  void *run_svm_sparse(void *data);
  void *run_lasso_one_sparse(void *data);
  void *run_svm_one_sparse(void *data);
  void *update_z_lasso_quantized(void *data);
  void *update_z_svm_quantized(void *data);    
  void *run_lasso_quantized(void *data);
  void *run_svm_quantized(void *data);
  
}

#endif
