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

#ifndef TASK_B_H
#define TASK_B_H

#include <cstdlib>
#include <cstdint>
#include <pthread.h>
#include <iostream>
#include <csignal>
#include <iomanip>
#include <algorithm>
#include <limits>

#include "threaded.h"
#include "algebra.h"
#include "vectorized.h"
#include "piece_pool.h"

#define B_USE_HBW true

class TaskB {
    
private:

  PiecePool* free_pool;
  SparseMatrix b_a_sparse;
  OneSparseMatrix b_a_one_sparse;
  QuantMatrix b_a_quant;
  Matrix b_a;              //subset of data
  #if HAS_QUANTIZED
  CloverVector32* b_acol32;
  #endif
    
  Vector b_b;              //subset of labels
  Vector b_norms;
  pthread_t* b_threads;    //threads
  bool* mark_a;            //should I update from A?
  bool* mark_b;            //should I update to B?
  uint32_t p;              //elements on B
  uint32_t n;              //features
  uint32_t d;              //samples
  uint32_t par_updates;
  uint32_t threads_per_vec;
  uint32_t atoms;
  uint32_t thread_size;
  real l;
  bool* a_running;          //used to stop A
  bool first_p_copy;        //first copy to B is different
  uint32_t* exe_order;
  struct threaded::SolverThreadArguments* thread_data;
  struct threaded::SolverArguments solver_data;
  bool no_chunk = false;
  
  pthread_attr_t attr;
  pthread_mutex_t cond_mutex;
  pthread_cond_t cond;
  pthread_mutex_t bar_mutex;
  uint32_t bar1;
  uint32_t bar2;
  bool can_start = false;
  bool threads_running = false;
  
  inline void initialize_vecs_common(bool primal);
  inline void initialize_vecs_common_one(bool primal);
  inline void initialize_vecs(bool primal);
  inline void initialize_vecs(OneSparseMatrix a, Vector b, bool primal);
  inline void initialize_vecs(QuantMatrix a, Vector b, bool primal);
  inline void initialize_vecs(Matrix a, Vector b, bool primal);
  
  inline void initialize_common_pre(uint32_t samples,
      uint32_t features, real lambda, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool* running, bool primal);
  inline void initialize_common_post(bool primal);

  inline void initialize_thread_data(bool primal);
  void shuffle();
  uint32_t find_idx_a(uint32_t* arr, uint32_t el_b, uint32_t p);
  
  void init_threads();
  void tStart();
  void tStop();

  std::default_random_engine random_engine;

public:
    
  uint32_t* b_p;
  Vector b_alpha;    //parameters
  Vector b_w;        //dual parameter
  
  void run_svm();
  void run_svm_sequential();
  void run_lasso();
  void run_lasso_sequential();
  void run_lasso_omp();
  void run_svm_omp();
  
  void initialize(uint32_t samples, uint32_t features, uint64_t pieces,
      uint32_t b_size, real lambda, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool* running, bool primal);

  void initialize_all(OneSparseMatrix a, Vector b, uint32_t samples,
      uint32_t features, real lambda, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool* running, bool primal);

  void initialize_all(QuantMatrix a, Vector b, uint32_t samples,
      uint32_t features, real lambda, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool* running, bool primal);

  void initialize_all(Matrix a, Vector b, uint32_t samples,
      uint32_t features, real lambda, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool* running, bool primal);

  void deinitialize();
  void deinitialize_ab();

  void update(uint32_t* a_p, OneSparseMatrix a, Vector b, Vector alpha,
      Vector norms, bool primal, uint32_t& copied);
  void update(uint32_t* a_p, QuantMatrix a, Vector b, Vector alpha,
      Vector norms, bool primal, uint32_t& copied);
  void update(uint32_t* a_p, Matrix a, Vector b, Vector alpha,
      Vector norms, bool primal, uint32_t& copied);

  uint32_t get_b_size();
  
  void join_threads();

};

#endif

