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

#ifndef TASK_A_H
#define TASK_A_H

#include <cstdlib>
#include <cstdint>
#include <pthread.h>
#include <algorithm>
#include <iomanip>
#include <iostream>

#include "algebra.h"
#include "threaded.h"
#include "vectorized.h"

#define A_USE_HBW false

class TaskA {

private:
   
  Vector a_z;                   //gaps
  Vector a_w;                   //dual parameters
  #if HAS_QUANTIZED
  CloverVector32* a_acol32;
  #endif
  pthread_t *a_threads;        //threads
  real l;                      //regularization parameter
  uint32_t d;                  //samples
  uint32_t n;                  //features
  uint32_t *exe_order;
  uint32_t par_updates;
  uint32_t b_thread_offset;
  uint32_t *updated_ctr;
  uint32_t total_updated;
  real bound;
  real duality_gap;
  real *a_z_copy;
  real objective;
  bool *a_running;
  bool b_only;
  class CompareIdxDesc;
  struct threaded::GapThreadArguments* thread_data;
  struct threaded::GapArguments gap_data;
  
  pthread_attr_t attr;
  pthread_mutex_t a_cond_mutex;
  pthread_cond_t a_cond;
  pthread_mutex_t a_bar_mutex;
  uint32_t a_bar1;
  uint32_t a_bar2;
  bool a_can_start = false;
  bool a_threads_running = false;
  
  std::minstd_rand0 random_engine;
  
  inline void initialize_vecs_common(Vector b, bool primal);
  inline void initialize_vecs(OneSparseMatrix a, Vector b, bool primal);
  inline void initialize_vecs(QuantMatrix a, Vector b, bool primal);
  inline void initialize_vecs(Matrix a, Vector b, bool primal);
  inline void initialize_thread_data(bool primal);
  inline void update_z_lasso(uint32_t index);
  inline void update_z_svm(uint32_t index);
  inline real partition(uint32_t size);
  inline void initialize_common_pre(uint32_t samples, uint32_t features,
      uint32_t b_elements, real lambda, uint32_t a_par_updates,
      uint32_t b_par_updates, uint32_t b_threads_per_vec,
      bool *running, bool primal, bool b_only);
  inline void initialize_common_post(bool primal);
  
  void init_threads();
  void tStart();
  void tStop();
    
public:

  OneSparseMatrix a_a_one_sparse;
  QuantMatrix a_a_quant;
  Matrix a_a;
  
  Vector a_b;      //labels
  Vector a_alpha;  //primal parameters
  Vector a_norms;  //A column norms
  uint32_t *a_p;   //data indices (first elements are transferred)
  uint32_t p;      //#elements on B

	void run_lasso_omp();
  void run_lasso();
  void run_lasso_sequential();
  void run_svm_omp();
  void run_svm();
  void run_svm_sequential();
  
  void initialize(OneSparseMatrix a, Vector b,
      uint32_t samples, uint32_t features, uint32_t b_elements,
      real lambda, uint32_t a_par_updates, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool *running, bool primal,
      bool b_only);

  void initialize(QuantMatrix a, Vector b,
      uint32_t samples, uint32_t features, uint32_t b_elements,
      real lambda, uint32_t a_par_updates, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool *running, bool primal,
      bool b_only);
      
  void initialize(Matrix a, Vector b,
      uint32_t samples, uint32_t features, uint32_t b_elements,
      real lambda, uint32_t a_par_updates, uint32_t b_par_updates,
      uint32_t b_threads_per_vec, bool *running, bool primal,
      bool b_only);
      
  void deinitialize();
  void update_p(bool primal);
  void update_alpha(Vector alpha, uint32_t *element_idx,
      uint32_t max_idx);
  void update_w(Vector w);
  void update_parameters(Vector alpha, Vector w);
  Vector get_weights(bool primal);
  void shuffle(bool primal);
  real get_duality_gap();
  real get_objective();
  real get_total_updated();
  void update_lasso_stats();
  void update_svm_stats();
  
  void join_threads();

};

#endif
