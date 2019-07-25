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

#include "threaded.h"

namespace threaded {
  
  struct barrier_wrapper {
    uint32_t bar;
    bool padding[60];
  };
  
  struct mutex_wrapper {
    pthread_mutex_t mutex;
    bool padding[24];
  };
  
  struct float_wrapper {
    float val;
    bool padding[60];
  };

  mutex_wrapper *b_sum_mutexes;
  mutex_wrapper *b_w_mutexes;
  barrier_wrapper *b_barrier_1;
  barrier_wrapper *b_barrier_2;
  barrier_wrapper *b_barrier_3;
  
  uint32_t reset_threads;
  uint32_t* a_seed;

  void initialize_b(uint32_t par_updates, uint32_t threads_per_vec,
      uint32_t atoms, uint32_t data_len, uint32_t p, bool use_hbw) {
    b_sum_mutexes = (mutex_wrapper*)b_malloc(par_updates
        * sizeof(mutex_wrapper), use_hbw);
    b_w_mutexes = (mutex_wrapper*)b_malloc(atoms
        * sizeof(mutex_wrapper), use_hbw);
    b_barrier_1 = (barrier_wrapper*)b_malloc(
        par_updates * sizeof(barrier_wrapper), use_hbw);
    b_barrier_2 = (barrier_wrapper*)b_malloc(
        par_updates * sizeof(barrier_wrapper), use_hbw);
    b_barrier_3 = (barrier_wrapper*)b_malloc(
        par_updates * sizeof(barrier_wrapper), use_hbw);
    for (uint32_t b = 0; b < par_updates; ++b) {
      pthread_mutex_init(&b_sum_mutexes[b].mutex, NULL);
      b_barrier_1[b].bar = threads_per_vec;
      b_barrier_2[b].bar = threads_per_vec;
      b_barrier_3[b].bar = threads_per_vec;
    }
    for (uint32_t a = 0; a < atoms; ++a)
      pthread_mutex_init(&b_w_mutexes[a].mutex, NULL);
    reset_threads = threads_per_vec;
    
  }

  void deinitialize_b(uint32_t par_updates, uint32_t threads_per_vec,
      uint32_t atoms, bool use_hbw) {
    for (uint32_t b = 0; b < par_updates; ++b)
      pthread_mutex_destroy(&b_sum_mutexes[b].mutex);
    for (uint32_t a = 0; a < atoms; ++a)
      pthread_mutex_destroy(&b_w_mutexes[a].mutex);
    b_free(b_barrier_1, use_hbw);
    b_free(b_barrier_2, use_hbw);
    b_free(b_barrier_3, use_hbw);
    b_free(b_sum_mutexes, use_hbw);
    b_free(b_w_mutexes, use_hbw);
  }
    
  void initialize_a(uint32_t par_updates, uint32_t columns,
      bool use_hbw) {
    a_seed = (uint32_t*)b_malloc(
        par_updates * sizeof(uint32_t), use_hbw);
    for (uint32_t b = 0; b < par_updates; ++b)
      a_seed[b]
          = std::chrono::system_clock::now().time_since_epoch().count();
  }
    
  void deinitialize_a(uint32_t par_updates, bool use_hbw) {
    b_free(a_seed, use_hbw);
  }
  
  inline uint32_t fastrand(uint32_t idx) {
    a_seed[idx] = (214013 * a_seed[idx] + 2531011) & 0xffffffff;
    return a_seed[idx];
  }
  
  inline bool no_shuffle(GapArguments* args) {
    return args->n_gaps > SHUFFLE_THRESHOLD && !args->b_only;
  }
  
  void run_lasso_at_index_dense(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    real tau, gamma, norm, product, sign, old_alpha, new_alpha, delta;
    real ln = args->regularization;    
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    Matrix b_a = args->b_a;
    Vector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;

      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0) {
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        tau = ln / norm;
        gamma = (old_alpha * norm - product) / norm;
        sign = (gamma == 0.0) ? 0.0 : (gamma > 0.0 ? 1.0 : -1.0);
        new_alpha = sign * std::max((real)0.0, (std::abs(gamma) - tau));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];
      
      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
          #if SCALAR
          scalar_multiply_add(b_w, b_a_col, delta, s, e);
          #else
          scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
          #endif
          pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }

  void *run_lasso_dense(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len);
    for (uint32_t step = start; step < end; ++step) {
      run_lasso_at_index_dense(arguments,
          arguments->args->exe_order[step], offset, range);
    }
  }

  void run_svm_at_index_dense(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    uint32_t n = args->data_len;
    real label, delta, norm, product, old_alpha, new_alpha;
    real ld = args->regularization;
   
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    Matrix b_a = args->b_a;
    Vector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;
  
      label = get_value(args->b_b, index);
      
      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0){
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        delta = (label - product / ld) / (norm / ld);
        new_alpha = label * std::max((real)0.0,
            std::min((real)1.0, label * (old_alpha + delta)));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];

      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
          #if SCALAR
          scalar_multiply_add(b_w, b_a_col, delta, s, e);
          #else
          scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
          #endif
          pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }
    
  void *run_svm_dense(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len); 
    for (uint32_t step = start; step < end; ++step)
      run_svm_at_index_dense(arguments,
          arguments->args->exe_order[step], offset, range);
  }


  void update_z_lasso_at_index_dense(struct GapThreadArguments *data,
      uint32_t index) {
    struct GapArguments *args = data->args;
    if (get_value(args->a_norms, index) == 0) {
      set_value(args->a_z, index, 0);
    } else {
      real alpha = get_value(args->a_alpha, index);
      #if SCALAR
      real product = dot_product(args->a_w,
          get_column(args->a_a, index));
      #else
      real product = dot_product_v(args->a_w,
          get_column(args->a_a, index));
      #endif
      real new_z = alpha * product
          + args->regularization * std::abs(alpha)
          + args->bound * std::max((real)0.0, std::abs(product)
          - args->regularization);
      set_value(args->a_z, index, new_z);
    }
  }
  
  void *update_z_lasso_dense(void* data) {
    struct GapThreadArguments *arguments
        = (struct GapThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->n_gaps
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->n_gaps,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t *exe_order = arguments->args->exe_order;
    if (no_shuffle(arguments->args)) {
      while (*(arguments->args->running)) {
        uint32_t idx = start + fastrand(arguments->update_thread_id)
            % (end - start);
        update_z_lasso_at_index_dense(arguments, idx);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    } else {
      if (start < end)
        std::shuffle(&exe_order[start], &exe_order[end],
            *(arguments->args->random_engine));
      for (uint32_t step = start; step < end; ++step) {
        if (!*(arguments->args->running))
          break;
        update_z_lasso_at_index_dense(arguments, exe_order[step]);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    }
  }

  void update_z_svm_at_index_dense(struct GapThreadArguments *data,
      uint32_t index) {
    struct GapArguments *args = data->args;
    if (get_value(args->a_norms, index) == 0) {
      set_value(args->a_z, index, 0);
    } else {
      real label = get_value(args->a_b, index);
      #if SCALAR
      real product = dot_product(args->a_w,
          get_column(args->a_a, index));
      #else
      real product = dot_product_v(args->a_w,
          get_column(args->a_a, index));
      #endif
      real score = product / (args->regularization);
      real alpha = get_value(args->a_alpha, index);
      real new_z = score * alpha
          + std::max((real)0.0, (real)1.0 - label * score)
          - label * alpha;
      set_value(args->a_z, index, new_z);
    }
  }
        
  void *update_z_svm_dense(void* data) {
    struct GapThreadArguments *arguments
        = (struct GapThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->n_gaps
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->n_gaps,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t *exe_order = arguments->args->exe_order;
    if (no_shuffle(arguments->args)) {
      while (*(arguments->args->running)) {
        uint32_t idx = start + fastrand(arguments->update_thread_id)
            % (end - start);
        update_z_svm_at_index_dense(arguments, idx);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    } else {
      if (start < end)
        std::shuffle(&exe_order[start], &exe_order[end],
            *(arguments->args->random_engine));
      for (uint32_t step = start; step < end; ++step) {
        if (!*(arguments->args->running))
          break;
        update_z_svm_at_index_dense(arguments, exe_order[step]);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    }
  }
  
  void run_lasso_at_index_sparse(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    real tau, gamma, norm, product, sign, old_alpha, new_alpha, delta;
    real ln = args->regularization;    
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    SparseMatrix b_a = args->b_a_sparse;
    SparseVector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;

      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0) {
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        tau = ln / norm;
        gamma = (old_alpha * norm - product) / norm;
        sign = (gamma == 0.0) ? 0.0 : (gamma > 0.0 ? 1.0 : -1.0);
        new_alpha = sign * std::max((real)0.0, (std::abs(gamma) - tau));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];
      
      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          if (b_a_col.max_idx >= s && b_a_col.data
              && b_a_col.data->indices[0] < e) {
            pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
            #if SCALAR
            scalar_multiply_add(b_w, b_a_col, delta, s, e);
            #else
            scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
            #endif
            pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
          }
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }

  void *run_lasso_sparse(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len);
    for (uint32_t step = start; step < end; ++step) {
      run_lasso_at_index_sparse(arguments,
          arguments->args->exe_order[step], offset, range);
    }
  }

  void run_svm_at_index_sparse(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    uint32_t n = args->data_len;
    real label, delta, norm, product, old_alpha, new_alpha;
    real ld = args->regularization;
   
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    SparseMatrix b_a = args->b_a_sparse;
    SparseVector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;
  
      label = get_value(args->b_b, index);
      
      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0){
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        delta = (label - product / ld) / (norm / ld);
        new_alpha = label * std::max((real)0.0,
            std::min((real)1.0, label * (old_alpha + delta)));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];

      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          if (b_a_col.max_idx >= s && b_a_col.data
              && b_a_col.data->indices[0] < e) {
            pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
            #if SCALAR
            scalar_multiply_add(b_w, b_a_col, delta, s, e);
            #else
            scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
            #endif
            pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
          }
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }
    
  void *run_svm_sparse(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len); 
    for (uint32_t step = start; step < end; ++step)
      run_svm_at_index_sparse(arguments,
          arguments->args->exe_order[step], offset, range);
  }


  void update_z_lasso_at_index_sparse(struct GapThreadArguments *data,
      uint32_t index) {
    struct GapArguments *args = data->args;
    if (get_value(args->a_norms, index) == 0) {
      set_value(args->a_z, index, 0);
    } else {
      real alpha = get_value(args->a_alpha, index);
      #if SCALAR
      real product = dot_product(args->a_w,
          get_column(args->a_a_one_sparse, index));
      #else
      real product = dot_product_v(args->a_w,
          get_column(args->a_a_one_sparse, index));
      #endif
      real new_z = alpha * product
          + args->regularization * std::abs(alpha)
          + args->bound * std::max((real)0.0, std::abs(product)
          - args->regularization);
      set_value(args->a_z, index, new_z);
    }
  }
  
  void *update_z_lasso_sparse(void* data) {
    struct GapThreadArguments *arguments
        = (struct GapThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->n_gaps
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->n_gaps,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t *exe_order = arguments->args->exe_order;
    if (no_shuffle(arguments->args)) {
      while (*(arguments->args->running)) {
        uint32_t idx = start + fastrand(arguments->update_thread_id)
            % (end - start);
        update_z_lasso_at_index_sparse(arguments, idx);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    } else {
      if (start < end)
        std::shuffle(&exe_order[start], &exe_order[end],
            *(arguments->args->random_engine));
      for (uint32_t step = start; step < end; ++step) {
        if (!*(arguments->args->running))
          break;
        update_z_lasso_at_index_sparse(arguments, exe_order[step]);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    }
  }

  void update_z_svm_at_index_sparse(struct GapThreadArguments *data,
      uint32_t index) {
    struct GapArguments *args = data->args;
    if (get_value(args->a_norms, index) == 0) {
      set_value(args->a_z, index, 0);
    } else {
      real label = get_value(args->a_b, index);
      #if SCALAR
      real product = dot_product(args->a_w,
          get_column(args->a_a_one_sparse, index));
      #else
      real product = dot_product_v(args->a_w,
          get_column(args->a_a_one_sparse, index));
      #endif
      real score = product / (args->regularization);
      real alpha = get_value(args->a_alpha, index);
      real new_z = score * alpha
          + std::max((real)0.0, (real)1.0 - label * score)
          - label * alpha;
      set_value(args->a_z, index, new_z);
    }
  }
        
  void *update_z_svm_sparse(void* data) {
    struct GapThreadArguments *arguments
        = (struct GapThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->n_gaps
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->n_gaps,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t *exe_order = arguments->args->exe_order;
    if (no_shuffle(arguments->args)) {
      while (*(arguments->args->running)) {
        uint32_t idx = start + fastrand(arguments->update_thread_id)
            % (end - start);
        update_z_svm_at_index_sparse(arguments, idx);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    } else {
      if (start < end)
        std::shuffle(&exe_order[start], &exe_order[end],
            *(arguments->args->random_engine));
      for (uint32_t step = start; step < end; ++step) {
        if (!*(arguments->args->running))
          break;
        update_z_svm_at_index_sparse(arguments, exe_order[step]);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    }
  }
  
  void run_lasso_at_index_one_sparse(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    real tau, gamma, norm, product, sign, old_alpha, new_alpha, delta;
    real ln = args->regularization;    
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    OneSparseMatrix b_a = args->b_a_one_sparse;
    OneSparseVector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;

      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0) {
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        tau = ln / norm;
        gamma = (old_alpha * norm - product) / norm;
        sign = (gamma == 0.0) ? 0.0 : (gamma > 0.0 ? 1.0 : -1.0);
        new_alpha = sign * std::max((real)0.0, (std::abs(gamma) - tau));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];
      
      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          if (b_a_col.max_idx >= s && b_a_col.indices[0] < e) {
            pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
            #if SCALAR
            scalar_multiply_add(b_w, b_a_col, delta, s, e);
            #else
            scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
            #endif
            pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
          }
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }

  void *run_lasso_one_sparse(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len);
    for (uint32_t step = start; step < end; ++step) {
      run_lasso_at_index_one_sparse(arguments,
          arguments->args->exe_order[step], offset, range);
    }
  }

  void run_svm_at_index_one_sparse(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    uint32_t n = args->data_len;
    real label, delta, norm, product, old_alpha, new_alpha;
    real ld = args->regularization;
   
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    OneSparseMatrix b_a = args->b_a_one_sparse;
    OneSparseVector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;
  
      label = get_value(args->b_b, index);
      
      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0){
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        delta = (label - product / ld) / (norm / ld);
        new_alpha = label * std::max((real)0.0,
            std::min((real)1.0, label * (old_alpha + delta)));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];

      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          if (b_a_col.max_idx >= s && b_a_col.indices[0] < e) {
            pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
            #if SCALAR
            scalar_multiply_add(b_w, b_a_col, delta, s, e);
            #else
            scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
            #endif
            pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
          }
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }
    
  void *run_svm_one_sparse(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len); 
    for (uint32_t step = start; step < end; ++step)
      run_svm_at_index_one_sparse(arguments,
          arguments->args->exe_order[step], offset, range);
  }
  
  void run_lasso_at_index_quantized(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    real tau, gamma, norm, product, sign, old_alpha, new_alpha, delta;
    real ln = args->regularization;    
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    QuantMatrix b_a = args->b_a_quant;
    QuantVector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;

      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      if (vec_thread_id == 0) {
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        tau = ln / norm;
        gamma = (old_alpha * norm - product) / norm;
        sign = (gamma == 0.0) ? 0.0 : (gamma > 0.0 ? 1.0 : -1.0);
        new_alpha = sign * std::max((real)0.0, (std::abs(gamma) - tau));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }
      
      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];
      
      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
          #if SCALAR
          scalar_multiply_add(b_w, b_a_col, delta, s, e);
          #else
          scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
          #endif
          pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }

  void *run_lasso_quantized(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len);
    for (uint32_t step = start; step < end; ++step) {
      run_lasso_at_index_quantized(arguments,
          arguments->args->exe_order[step], offset, range);
    }
  }

  void run_svm_at_index_quantized(struct SolverThreadArguments *data,
      uint32_t index, uint32_t offset, uint32_t range) {
    struct SolverArguments *args = data->args;
    uint32_t vec_thread_id = data->vec_thread_id;
    uint32_t update_thread_id = data->update_thread_id;
    real *alpha_diffs = args->alpha_diffs;
    real *prods = args->prods;
    uint32_t n = args->data_len;
    real label, delta, norm, product, old_alpha, new_alpha;
    real ld = args->regularization;
   
    Vector b_alpha = args->b_alpha;
    Vector b_norms = args->b_norms;
    QuantMatrix b_a = args->b_a_quant;
    QuantVector b_a_col;
    Vector b_w = args->b_w;
    
    norm = get_value(b_norms, index);
    if (norm == 0) {
      if (vec_thread_id == 0)
        set_value(b_alpha, index, 0);
    } else {
      b_a_col = get_column(b_a, index);
      if (vec_thread_id == 0)
        prods[update_thread_id] = 0;

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_1[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_1[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0)
        b_barrier_3[update_thread_id].bar = reset_threads;
  
      label = get_value(args->b_b, index);
      
      #if SCALAR
      product = dot_product(b_w, b_a_col, offset, range);
      #else
      product = dot_product_v(b_w, b_a_col, offset, range);
      #endif

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      prods[update_thread_id] += product;
      --b_barrier_2[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_2[update_thread_id].bar != 0) nsleep(SLEEP_TIME);

      if (vec_thread_id == 0){
        product = prods[update_thread_id];
        old_alpha = get_value(b_alpha, index);
        delta = (label - product / ld) / (norm / ld);
        new_alpha = label * std::max((real)0.0,
            std::min((real)1.0, label * (old_alpha + delta)));
        alpha_diffs[update_thread_id] = new_alpha - old_alpha;
        b_barrier_1[update_thread_id].bar = reset_threads;
      }

      pthread_mutex_lock(&b_sum_mutexes[update_thread_id].mutex);
      --b_barrier_3[update_thread_id].bar;
      pthread_mutex_unlock(&b_sum_mutexes[update_thread_id].mutex);
      while (b_barrier_3[update_thread_id].bar != 0) nsleep(SLEEP_TIME);
      
      delta = alpha_diffs[update_thread_id];

      if (std::abs(delta) > 0) {
        #if LOCK
        uint32_t chunk = B_THREAD_CHUNK_SIZE;
        for (uint32_t i = offset; i < range; i += chunk) {
          uint32_t s = i;
          uint32_t e = std::min(i + chunk, range);
          uint32_t lock_idx = i / chunk;
          pthread_mutex_lock(&b_w_mutexes[lock_idx].mutex);
          #if SCALAR
          scalar_multiply_add(b_w, b_a_col, delta, s, e);
          #else
          scalar_multiply_add_v(b_w, b_a_col, delta, s, e);
          #endif
          pthread_mutex_unlock(&b_w_mutexes[lock_idx].mutex);
        }
        #else
        #if SCALAR
        scalar_multiply_add(b_w, b_a_col, delta, offset, range);
        #else
        scalar_multiply_add_v(b_w, b_a_col, delta, offset, range);
        #endif
        #endif
      }
      if (vec_thread_id == 0) {
        b_barrier_2[update_thread_id].bar = reset_threads;
        if(std::abs(delta) > 0)
          set_value(b_alpha, index, new_alpha);
      }
    }
  }
    
  void *run_svm_quantized(void* data) {
    struct SolverThreadArguments *arguments
        = (struct SolverThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->b_size
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->b_size,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t offset = arguments->vec_thread_id
        * arguments->args->thread_size;
    uint32_t range = std::min(offset + arguments->args->thread_size,
        arguments->args->data_len); 
    for (uint32_t step = start; step < end; ++step)
      run_svm_at_index_quantized(arguments,
          arguments->args->exe_order[step], offset, range);
  }


  void update_z_lasso_at_index_quantized(
      struct GapThreadArguments *data, uint32_t index) {
    struct GapArguments *args = data->args;
    if (get_value(args->a_norms, index) == 0) {
      set_value(args->a_z, index, 0);
    } else {
      real alpha = get_value(args->a_alpha, index);
      #if SCALAR
      real product = dot_product(args->a_w,
          get_column(args->a_a_quant, index));
      #else
      real product = dot_product_v(args->a_w,
          get_column(args->a_a_quant, index));
      #endif
      real new_z = alpha * product
          + args->regularization * std::abs(alpha)
          + args->bound * std::max((real)0.0, std::abs(product)
          - args->regularization);
      set_value(args->a_z, index, new_z);
    }
  }
  
  void *update_z_lasso_quantized(void* data) {
    struct GapThreadArguments *arguments
        = (struct GapThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->n_gaps
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->n_gaps,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t *exe_order = arguments->args->exe_order;
    if (no_shuffle(arguments->args)) {
      while (*(arguments->args->running)) {
        uint32_t idx = start + fastrand(arguments->update_thread_id)
            % (end - start);
        update_z_lasso_at_index_quantized(arguments, idx);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    } else {
      if (start < end)
        std::shuffle(&exe_order[start], &exe_order[end],
            *(arguments->args->random_engine));
      for (uint32_t step = start; step < end; ++step) {
        if (!*(arguments->args->running))
          break;
        update_z_lasso_at_index_quantized(arguments, exe_order[step]);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    }
  }

  void update_z_svm_at_index_quantized(struct GapThreadArguments *data,
      uint32_t index) {
    struct GapArguments *args = data->args;
    if (get_value(args->a_norms, index) == 0) {
      set_value(args->a_z, index, 0);
    } else {
      real label = get_value(args->a_b, index);
      #if SCALAR
      real product = dot_product(args->a_w,
          get_column(args->a_a_quant, index));
      #else
      real product = dot_product_v(args->a_w,
          get_column(args->a_a_quant, index));
      #endif
      real score = product / (args->regularization);
      real alpha = get_value(args->a_alpha, index);
      real new_z = score * alpha
          + std::max((real)0.0, (real)1.0 - label * score)
          - label * alpha;
      set_value(args->a_z, index, new_z);
    }
  }
        
  void *update_z_svm_quantized(void* data) {
    struct GapThreadArguments *arguments
        = (struct GapThreadArguments*)data;
    uint32_t col_range = std::ceil((real)arguments->args->n_gaps
        / (real)arguments->args->par_updates);
    uint32_t start = arguments->update_thread_id * col_range;
    uint32_t end = std::min(arguments->args->n_gaps,
        (arguments->update_thread_id + 1) * col_range);
    uint32_t *exe_order = arguments->args->exe_order;
    if (no_shuffle(arguments->args)) {
      while (*(arguments->args->running)) {
        uint32_t idx = start + fastrand(arguments->update_thread_id)
            % (end - start);
        update_z_svm_at_index_quantized(arguments, idx);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    } else {
      if (start < end)
        std::shuffle(&exe_order[start], &exe_order[end],
            *(arguments->args->random_engine));
      for (uint32_t step = start; step < end; ++step) {
        if (!*(arguments->args->running))
          break;
        update_z_svm_at_index_quantized(arguments, exe_order[step]);
        ++(arguments->args->updated[arguments->update_thread_id]);
      }
    }
  }

}
