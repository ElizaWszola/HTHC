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

#include "task_b.h"

inline void TaskB::initialize_vecs_common(bool primal) {
  b_w = create_vector(primal ? d : n, B_USE_HBW);
  b_alpha = create_vector(p, B_USE_HBW);
  b_norms = create_vector(p, B_USE_HBW);
  b_p = (uint32_t*)b_malloc(p * sizeof(uint32_t), B_USE_HBW);
  mark_a = (bool*)b_malloc(p * sizeof(bool), B_USE_HBW);
  mark_b = (bool*)b_malloc(p * sizeof(bool), B_USE_HBW);
  first_p_copy = true;
  exe_order = (uint32_t*)b_malloc(p * sizeof(uint32_t), B_USE_HBW);
  for (uint32_t i = 0; i < p; ++i)
    exe_order[i] = i;
}

inline void TaskB::initialize_vecs_common_one(bool primal) {
  if (primal) {
    #if SCALAR
    scalar_multiply(b_w, b_b, -1.0);
    #else
    scalar_multiply_v(b_w, b_b, -1.0);
    #endif
    if (data_rep == QUANTIZED)
      inner_transpose(b_w);
  }
  for (uint32_t i = 0; i < p; ++i)
    set_value(b_alpha, i, 0);
}

inline void TaskB::initialize_vecs(bool primal) {
  initialize_vecs_common(primal);
  if (primal) {
    if (data_rep == DENSE32)
      b_a = create_matrix(d, p, B_USE_HBW);
    else if (data_rep == SPARSE32)
      b_a_sparse = create_matrix_sparse(d, p, B_USE_HBW);
    else if (data_rep == QUANTIZED)
      b_a_quant = create_matrix_quant(d, p, B_USE_HBW);
    b_b = create_vector(d, B_USE_HBW);
  } else {
    if (data_rep == DENSE32)
      b_a = create_matrix(n, p, B_USE_HBW);
    else if (data_rep == SPARSE32)
      b_a_sparse = create_matrix_sparse(n, p, B_USE_HBW);
    else if (data_rep == QUANTIZED)
      b_a_quant = create_matrix_quant(n, p, B_USE_HBW);
    b_b = create_vector(p, B_USE_HBW);
  }
}

inline void TaskB::initialize_vecs(OneSparseMatrix a, Vector b,
    bool primal) {
  b_a_one_sparse = a;
  b_b = b;
  initialize_vecs_common(primal);
  initialize_vecs_common_one(primal);
  for (uint32_t i = 0; i < p; ++i) {
    #if SCALAR
    set_value(b_norms, i, norm_2_squared(get_column(a, i)));
    #else
    set_value(b_norms, i, norm_2_squared_v(get_column(a, i)));
    #endif
  }
}

inline void TaskB::initialize_vecs(QuantMatrix a, Vector b,
    bool primal) {
  b_a_quant = a;
  b_b = b;
  initialize_vecs_common(primal);
  initialize_vecs_common_one(primal);
  for (uint32_t i = 0; i < p; ++i) {
    #if SCALAR
    set_value(b_norms, i, norm_2_squared(get_column(a, i)));
    #else
    set_value(b_norms, i, norm_2_squared_v(get_column(a, i)));
    #endif
  }
}

inline void TaskB::initialize_vecs(Matrix a, Vector b, bool primal) {
  b_a = a;
  b_b = b;
  initialize_vecs_common(primal);
  initialize_vecs_common_one(primal);
  for (uint32_t i = 0; i < p; ++i) {
    #if SCALAR
    set_value(b_norms, i, norm_2_squared(get_column(a, i)));
    #else
    set_value(b_norms, i, norm_2_squared_v(get_column(a, i)));
    #endif
  }
}

inline void TaskB::initialize_thread_data(bool primal) {
  solver_data.b_alpha = b_alpha;
  solver_data.b_w = b_w;
  solver_data.b_b = b_b;
  solver_data.b_norms = b_norms;
  solver_data.b_a = b_a;
  solver_data.b_a_sparse = b_a_sparse;
  solver_data.b_a_one_sparse = b_a_one_sparse;
  solver_data.b_a_quant = b_a_quant;
  solver_data.b_a = b_a;
  solver_data.regularization = l * d;
  solver_data.data_len = primal ? d : n;
  solver_data.thread_size = thread_size;
  solver_data.exe_order = exe_order;
  solver_data.threads_per_vec = threads_per_vec;
  solver_data.par_updates = par_updates;
  solver_data.b_size = p;
  solver_data.prods = (real*)b_malloc(par_updates * sizeof(real),
      B_USE_HBW);
  solver_data.alpha_diffs = (real*)b_malloc(par_updates * sizeof(real),
      B_USE_HBW);
  std::memset(solver_data.prods, 0, par_updates * sizeof(real));
  thread_data = (threaded::SolverThreadArguments*)
      b_malloc(threads_per_vec * par_updates
          * sizeof(threaded::SolverThreadArguments), B_USE_HBW);
  #if HAS_QUANTIZED
  solver_data.b_acol32 = b_acol32;
  #endif
  solver_data.cond_mutex = &cond_mutex;
  solver_data.cond = &cond;
  solver_data.bar_mutex = &bar_mutex;
  solver_data.bar1 = &bar1;
  solver_data.bar2 = &bar2;
  solver_data.can_start = &can_start;
  solver_data.threads_running = &threads_running;
  solver_data.primal = primal;
  solver_data.no_chunk = no_chunk;
}

inline void TaskB::initialize_common_pre(uint32_t samples,
    uint32_t features, real lambda, uint32_t b_par_updates,
    uint32_t b_threads_per_vec, bool* running, bool primal) {
  d = samples;
  n = features;
  l = lambda;
  real data_len = primal ? d : n;
  a_running = running;
  par_updates = b_par_updates;
  atoms = std::ceil(data_len / real(B_THREAD_CHUNK_SIZE));
  thread_size = uint32_t(B_THREAD_CHUNK_SIZE
      * std::ceil(atoms / (real)b_threads_per_vec));
  threads_per_vec = std::ceil(data_len / (real)thread_size);
  b_threads = (pthread_t*)b_malloc(threads_per_vec * par_updates
      * sizeof(pthread_t), B_USE_HBW);
  random_engine = std::default_random_engine(
      std::chrono::system_clock::now().time_since_epoch().count());
}

inline void TaskB::initialize_common_post(bool primal) {
  initialize_thread_data(primal);
  threaded::initialize_b(par_updates, threads_per_vec, atoms,
      primal ? d : n, p, B_USE_HBW);
  init_threads();
}

void TaskB::initialize(uint32_t samples, uint32_t features,
    uint64_t pieces, uint32_t b_size, real lambda,
    uint32_t b_par_updates, uint32_t b_threads_per_vec, bool* running,
    bool primal) {
  initialize_common_pre(samples, features, lambda,
      b_par_updates, b_threads_per_vec, running, primal);
  p = b_size;
  if (data_rep == SPARSE32) {
    free_pool = new PiecePool();
    free_pool->allocate(pieces, B_USE_HBW);
  }
  no_chunk = false;
  #if HAS_QUANTIZED
  if (data_rep == QUANTIZED) {
    b_acol32 = (CloverVector32*)b_malloc(
        par_updates * sizeof(CloverVector32), B_USE_HBW);
    for (uint32_t i = 0; i < par_updates; ++i)
      new(b_acol32 + i) CloverVector32(primal ? d : n);
  }
  #endif
  initialize_vecs(primal);
  initialize_common_post(primal);
}

void TaskB::initialize_all(OneSparseMatrix a, Vector b,
    uint32_t samples, uint32_t features, real lambda,
    uint32_t b_par_updates, uint32_t b_threads_per_vec, bool* running,
    bool primal) {
  initialize_common_pre(samples, features, lambda, b_par_updates,
      b_threads_per_vec, running, primal);
  p = primal ? n : d;
  initialize_vecs(a, b, primal);
  free_pool = nullptr;
  no_chunk = true;
  initialize_common_post(primal);
}

void TaskB::initialize_all(QuantMatrix a, Vector b, uint32_t samples,
    uint32_t features, real lambda, uint32_t b_par_updates,
    uint32_t b_threads_per_vec, bool* running, bool primal) {
  initialize_common_pre(samples, features, lambda, b_par_updates,
      b_threads_per_vec, running, primal);
  p = primal ? n : d;
  initialize_vecs(a, b, primal);
  #if HAS_QUANTIZED
  b_acol32 = (CloverVector32*)b_malloc(
      par_updates * sizeof(CloverVector32), B_USE_HBW);
  for (uint32_t i = 0; i < par_updates; ++i)
    new(b_acol32 + i) CloverVector32(primal ? d : n);
  #endif
  free_pool = nullptr;
  no_chunk = true;
  initialize_common_post(primal);
}

void TaskB::initialize_all(Matrix a, Vector b, uint32_t samples,
    uint32_t features, real lambda, uint32_t b_par_updates,
    uint32_t b_threads_per_vec, bool* running, bool primal) {
  initialize_common_pre(samples, features, lambda, b_par_updates,
      b_threads_per_vec, running, primal);
  p = primal ? n : d;
  initialize_vecs(a, b, primal);
  free_pool = nullptr;
  no_chunk = true;
  initialize_common_post(primal);
}

void TaskB::deinitialize() {
  join_threads();
  if (data_rep == SPARSE32) {
    if (free_pool)
      free_pool->deallocate(B_USE_HBW);
    delete free_pool;
  }
  destroy(b_w, B_USE_HBW);
  destroy(b_alpha, B_USE_HBW);
  destroy(b_norms, B_USE_HBW);
  #if HAS_QUANTIZED
  if (data_rep == QUANTIZED)
    b_free(b_acol32, B_USE_HBW);
  #endif
  b_free(b_p, B_USE_HBW);
  b_free(mark_a, B_USE_HBW);
  b_free(mark_b, B_USE_HBW);
  b_free(b_threads, B_USE_HBW);
  b_free(exe_order, B_USE_HBW);
  b_free(solver_data.alpha_diffs, B_USE_HBW);
  b_free(solver_data.prods, B_USE_HBW);
  b_free(thread_data, B_USE_HBW);
  threaded::deinitialize_b(par_updates, threads_per_vec, atoms,
      B_USE_HBW);
}

void TaskB::deinitialize_ab() {
  if (data_rep == DENSE32)
    destroy(b_a, B_USE_HBW);
  else if (data_rep == SPARSE32)
    destroy(b_a_sparse, B_USE_HBW);
  else if (data_rep == QUANTIZED)
    destroy(b_a_quant, B_USE_HBW);
  destroy(b_b, B_USE_HBW);
}

void TaskB::shuffle() {
  std::shuffle(&exe_order[0], &exe_order[p], random_engine);
}

void nsleep() {
  struct timespec tim;
  tim.tv_sec = 0;
  tim.tv_nsec = 0;
  nanosleep(&tim, NULL);
}

void* optimize(void* data) {
  struct threaded::SolverArguments *args
      = ((struct threaded::SolverThreadArguments*)data)->args;
  while (*(args->threads_running)) {
    pthread_mutex_lock(args->cond_mutex);
    if (!*(args->can_start))
      pthread_cond_wait(args->cond, args->cond_mutex);
    pthread_mutex_unlock(args->cond_mutex);
    if (*(args->threads_running)) {
      if (data_rep == DENSE32) {
        if(args->primal)
          threaded::run_lasso_dense(data);
        else
          threaded::run_svm_dense(data);
      } else if (data_rep == SPARSE32) {
        if(args->primal) {
          if (args->no_chunk)
            threaded::run_lasso_one_sparse(data);
          else
            threaded::run_lasso_sparse(data);
        } else {
          if (args->no_chunk)
            threaded::run_svm_one_sparse(data);
          else
            threaded::run_svm_sparse(data);
        }
      } else if (data_rep == QUANTIZED) {
        if(args->primal)
          threaded::run_lasso_quantized(data);
        else
          threaded::run_svm_quantized(data);
        
      }
      pthread_mutex_lock(args->bar_mutex);
      (*(args->bar1))++;
      pthread_mutex_unlock(args->bar_mutex);
      while (*(args->bar1) != 0) nsleep();
      pthread_mutex_lock(args->bar_mutex);
      (*(args->bar2))++;
      pthread_mutex_unlock(args->bar_mutex);
      while (*(args->bar2) != 0) nsleep();
    }
  }
  pthread_exit(0);
}
  
void TaskB::init_threads() {
  can_start = false;
  pthread_cond_init(&cond, NULL);
  pthread_mutex_init(&cond_mutex, NULL);
  pthread_mutex_init(&bar_mutex, NULL);
  threads_running = true;
  pthread_attr_init(&attr);
  cpu_set_t cpuset;
  uint32_t all_threads = threads_per_vec * par_updates;
  for (uint32_t b = 0; b < par_updates; ++b) {
    uint32_t row = b * threads_per_vec;
    for (uint32_t t = 0; t < threads_per_vec; ++t) {
      uint32_t bt = row + t;
      uint32_t affinity = TILE_MULTIPLIER * b
          + t * TILE_MULTIPLIER * par_updates;
      CPU_ZERO(&cpuset);
      CPU_SET(affinity, &cpuset);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
      thread_data[bt].vec_thread_id = t;
      thread_data[bt].update_thread_id = b;
      thread_data[bt].args = &solver_data;
      pthread_create(&b_threads[bt], &attr, optimize,
          (void*)&thread_data[bt]);
    }
  }
  bar1 = 0;
  bar2 = 0;
}

void TaskB::join_threads() {
  threads_running = false;
  tStart();
  uint32_t all_threads = threads_per_vec * par_updates;
  for (uint32_t bt = 0; bt < all_threads; ++bt)
    pthread_join(b_threads[bt], NULL);
  pthread_attr_destroy(&attr);
  pthread_cond_destroy(&cond);
  pthread_mutex_destroy(&cond_mutex);
  pthread_mutex_destroy(&bar_mutex);
}

void TaskB::tStart() {
  pthread_mutex_lock(&cond_mutex);
  pthread_cond_broadcast(&cond);
  can_start = true;
  pthread_mutex_unlock(&cond_mutex);
}

void TaskB::tStop() {
  can_start = false;
}

void TaskB::run_lasso() {
  shuffle();
  uint32_t all_threads = threads_per_vec * par_updates;
  tStart();
  while (bar1 < all_threads) nsleep();
  bar1 = 0;
  tStop();
  while (bar2 < all_threads) nsleep();
  bar2 = 0;
  for(uint32_t i=0; i<0; ++i)
    nsleep();
  *a_running = false;
}

void TaskB::run_svm() {
  shuffle();
  uint32_t all_threads = threads_per_vec * par_updates;
  tStart();
  while (bar1 < all_threads) nsleep();
  bar1 = 0;
  tStop();
  while (bar2 < all_threads) nsleep();
  bar2 = 0;
  for(uint32_t i=0; i<0; ++i)
    nsleep();
  *a_running = false;
}

//arr has to be sorted
//if p is returned, element not found
uint32_t TaskB::find_idx_a(uint32_t* arr, uint32_t el_b, uint32_t p) {
  uint32_t left = 0;
  uint32_t right = p;
  uint32_t mid;
  uint32_t el_a;
  if (el_b < arr[0] || el_b > arr[p - 1])
    return p;
  while (left < right) {
    mid = (left + right) / 2;
    el_a = arr[mid];
    if (el_a == el_b)
      return mid;
    else if (el_a < el_b)
      left = mid + 1;
    else
      right = mid;
  }
  if (left < p && arr[left] == el_b)
    return left;
  return p;
}

void TaskB::update(uint32_t* a_p, OneSparseMatrix a, Vector b,
  Vector alpha, Vector norms, bool primal, uint32_t& copied) {
    
  OneSparseVector cola;
  SparseVector colb;
  SparsePiece *piece;
  uint32_t copied_p = 0;
  solver_data.b_size = p;
  
  if (first_p_copy) {
    
    std::memcpy(b_p, a_p, p * sizeof(uint32_t));
    
    copied = p;
    //redistribute from the pool
    for (uint32_t i = 0; i < p; ++i) {
      uint32_t idx = b_p[i];
      cola = get_column(a, idx);
      uint32_t pieces_needed = std::ceil(
          (double)cola.padded_nnz / SPARSE_PIECE_LENGTH);
      b_a_sparse.column_data[i].data = nullptr;
      for (uint32_t j = 0; j < pieces_needed; ++j) {
        piece = free_pool->pop();
        piece->next = b_a_sparse.column_data[i].data;
        b_a_sparse.column_data[i].data = piece;
      }
      set_column(b_a_sparse, i, cola);
      set_value(b_norms, i, get_value(norms, idx));
      set_value(b_alpha, i, get_value(alpha, idx));
    }
    
    if (primal) {
      set(b_b, b);
    } else {
      for (uint32_t i = 0; i < p; ++i)
        set_value(b_b, i, get_value(b, b_p[i]));
    }
    if (first_p_copy) {
      if (primal) {
        #if SCALAR
        scalar_multiply(b_w, b, -1);
        #else
        scalar_multiply_v(b_w, b, -1);
        #endif
      } else {
        set_zero(b_w);
      }
    }
    first_p_copy = false;
    
  } else {
    
    copied = 0;
    memset(mark_a, 0, sizeof(bool) * p);
    memset(mark_b, 0, sizeof(bool) * p);
    uint32_t idx_a = 0;
    uint32_t idx_b = 0;
    std::sort(a_p, a_p + p);
    for (idx_b = 0; idx_b < p; ++idx_b) {
      idx_a = find_idx_a(a_p, b_p[idx_b], p);
      if (idx_a < p) {
        //mark idx not to be updated
        mark_a[idx_a] = true;
        mark_b[idx_b] = true;
      } else {
        //release to the pool
        colb = get_column(b_a_sparse, idx_b);
        while (colb.data) {
          piece = colb.data;
          colb.data = colb.data->next;
          piece->next = nullptr;
          free_pool->push(piece);
        }
        b_a_sparse.column_data[idx_b].data = nullptr;
      }
    }
    
    idx_a = 0;
    idx_b = 0;
    while (idx_a < p) { //perform updates
      if (!mark_a[idx_a]) {
        while (mark_b[idx_b])
          ++idx_b;
        uint32_t idx = a_p[idx_a];
        b_p[idx_b] = idx;
        cola = get_column(a, idx);
        uint32_t pieces_needed = std::ceil(
            (double)cola.padded_nnz / SPARSE_PIECE_LENGTH);
        b_a_sparse.column_data[idx_b].data = nullptr;
        for (uint32_t j = 0; j < pieces_needed; ++j) {
          piece = free_pool->pop();
          piece->next = b_a_sparse.column_data[idx_b].data;
          b_a_sparse.column_data[idx_b].data = piece;
        }
        set_column(b_a_sparse, idx_b, cola);
        set_value(b_alpha, idx_b, get_value(alpha, idx));
        set_value(b_norms, idx_b, get_value(norms, idx));
        if (!primal)
          set_value(b_b, idx_b, get_value(b, idx));
        ++idx_b;
        ++copied;
      }
      ++idx_a;
    }
  }
}

void TaskB::update(uint32_t* a_p, QuantMatrix a, Vector b, Vector alpha,
    Vector norms, bool primal, uint32_t& copied) {
  if (first_p_copy) {
    std::memcpy(b_p, a_p, p * sizeof(uint32_t));
    for (uint32_t i = 0; i < p; ++i) {
      uint32_t idx = b_p[i];
      set_column(b_a_quant, i, get_column(a, idx));
      set_value(b_norms, i, get_value(norms, idx));
      set_value(b_alpha, i, get_value(alpha, idx));
    }
    if (primal) {
      set(b_b, b);
      #if SCALAR
      scalar_multiply(b_w, b, -1);
      #else
      scalar_multiply_v(b_w, b, -1);
      #endif
      inner_transpose(b_w);
    } else {
      for (uint32_t i = 0; i < p; ++i)
        set_value(b_b, i, get_value(b, b_p[i]));
      set_zero(b_w);
    }
    first_p_copy = false;
  } else {
    copied = 0;
    memset(mark_a, 0, sizeof(bool) * p);
    memset(mark_b, 0, sizeof(bool) * p);
    uint32_t idx_a = 0;
    uint32_t idx_b = 0;
    std::sort(a_p, a_p + p);
    for (idx_b = 0; idx_b < p; ++idx_b) { //mark idx not to be updated
      idx_a = find_idx_a(a_p, b_p[idx_b], p);
      if (idx_a < p) {
        mark_a[idx_a] = true;
        mark_b[idx_b] = true;
      }
    }
    idx_a = 0;
    idx_b = 0;
    while (idx_a < p) { //perform updates
      if (!mark_a[idx_a]) {
        while (mark_b[idx_b])
          ++idx_b;
        uint32_t idx = a_p[idx_a];
        b_p[idx_b] = idx;
        set_column(b_a_quant, idx_b, get_column(a, idx));
        set_value(b_alpha, idx_b, get_value(alpha, idx));
        set_value(b_norms, idx_b, get_value(norms, idx));
        if (!primal)
          set_value(b_b, idx_b, get_value(b, idx));
        ++idx_b;
        ++copied;
      }
      ++idx_a;
    }
  }
    
}

void TaskB::update(uint32_t* a_p, Matrix a, Vector b, Vector alpha,
  Vector norms, bool primal, uint32_t& copied) {
    
  if (first_p_copy) {
    std::memcpy(b_p, a_p, p * sizeof(uint32_t));
    for (uint32_t i = 0; i < p; ++i) {
      uint32_t idx = b_p[i];
      set_column(b_a, i, get_column(a, idx));
      set_value(b_norms, i, get_value(norms, idx));
      set_value(b_alpha, i, get_value(alpha, idx));
    }
    if (primal) {
      set(b_b, b);
      #if SCALAR
      scalar_multiply(b_w, b, -1);
      #else
      scalar_multiply_v(b_w, b, -1);
      #endif
    } else {
      for (uint32_t i = 0; i < p; ++i)
        set_value(b_b, i, get_value(b, b_p[i]));
      set_zero(b_w);
    }
    first_p_copy = false;
  } else {
    copied = 0;
    memset(mark_a, 0, sizeof(bool) * p);
    memset(mark_b, 0, sizeof(bool) * p);
    uint32_t idx_a = 0;
    uint32_t idx_b = 0;
    std::sort(a_p, a_p + p);
    for (idx_b = 0; idx_b < p; ++idx_b) { //mark idx not to be updated
      idx_a = find_idx_a(a_p, b_p[idx_b], p);
      if (idx_a < p) {
        mark_a[idx_a] = true;
        mark_b[idx_b] = true;
      }
    }
    idx_a = 0;
    idx_b = 0;
    while (idx_a < p) { //perform updates
      if (!mark_a[idx_a]) {
        while (mark_b[idx_b])
          ++idx_b;
        uint32_t idx = a_p[idx_a];
        b_p[idx_b] = idx;
        set_column(b_a, idx_b, get_column(a, idx));
        set_value(b_alpha, idx_b, get_value(alpha, idx));
        set_value(b_norms, idx_b, get_value(norms, idx));
        if (!primal)
          set_value(b_b, idx_b, get_value(b, idx));
        ++idx_b;
        ++copied;
      }
      ++idx_a;
    }
  }
}

uint32_t TaskB::get_b_size() {
  return solver_data.b_size;
}

//These functions are only for comparison against the OMP baseline
//To disable atomics, comment out #pragma omp atomic and recompile
//The comparisons work only for dense data

void TaskB::run_lasso_omp() {
  shuffle();
  #pragma omp parallel for num_threads(par_updates)
  for (uint32_t s = 0; s < p; ++s) {
    uint32_t index = exe_order[s];
    real norm = get_value(b_norms, index);
    if (norm == 0) {
      set_value(b_alpha, index, 0);
    } else {
      real* a_col = get_column(b_a, index).data;
      real* w = b_w.data;
      real product = 0;
      #pragma omp simd reduction(+: product)
      for (uint32_t i = 0; i < d; ++i)
        product += a_col[i] * w[i];
      real old_alpha = get_value(b_alpha, index);
      real ln = l * d;
      real tau = ln / norm;
      real gamma = (old_alpha * norm - product) / norm;
      real sign = (gamma == 0.0) ? 0.0 : (gamma > 0.0 ? 1.0 : -1.0);
      real new_alpha = sign
          * std::max((real)0.0, (std::abs(gamma) - tau));
      new_alpha = old_alpha + (new_alpha - old_alpha);
      real alpha_diff = new_alpha - old_alpha;
      set_value(b_alpha, index, new_alpha);
      #pragma omp parallel for num_threads(threads_per_vec)
      for (uint32_t i = 0; i < d; ++i) {
        #pragma omp atomic
        w[i] = w[i] + a_col[i] * alpha_diff;
      }
    }
  }
  *a_running = false;
}

void TaskB::run_svm_omp() {
  shuffle();
  #pragma omp parallel for num_threads(par_updates)
  for (uint32_t s = 0; s < p; ++s) {
    uint32_t index = exe_order[s];
    real norm = get_value(b_norms, index);
    if (norm == 0) {
      set_value(b_alpha, index, 0);
    } else {
      real label = get_value(b_b, index);
      real* a_col = get_column(b_a, index).data;
      real* w = b_w.data;
      real product = 0;
      #pragma omp simd reduction(+: product)
      for (uint32_t i = 0; i < n; ++i)
        product += a_col[i] * w[i];
      real old_alpha = get_value(b_alpha, index);
      real ln = l * d;
      real delta = (label - product / ln) / (norm / ln);
      real new_alpha = label * std::max((real)0.0,
          std::min((real)1.0, label * (old_alpha + delta)));
      real alpha_diff = new_alpha - old_alpha;
      set_value(b_alpha, index, new_alpha);
      #pragma omp parallel for num_threads(threads_per_vec)
      for (uint32_t i = 0; i < n; ++i) {
        #pragma omp atomic
        w[i] = w[i] + a_col[i] * alpha_diff;
      }
    }
  }
  *a_running = false;
}

//sequential updates - these functions are left here for troubleshooting
//to use, replace run_lasso() and run_svm() calls in main.cpp
void TaskB::run_lasso_sequential() {
  real tau, gamma, norm, product, sign, old_alpha, new_alpha;
  real ln = l * d;
  uint32_t i;
  SparseVector a_col_sparse;
  QuantVector a_col_quant;
  Vector a_col;
  for (uint32_t s = 0; s < p; ++s) {
    i = s;
    a_col = get_column(b_a, i);
    norm = get_value(b_norms, i);
    if (norm == 0) {
      set_value(b_alpha, i, 0);
    } else {
      old_alpha = get_value(b_alpha, i);
      if (data_rep == DENSE32) {
        a_col = get_column(b_a, i);
        #if SCALAR
        product = dot_product(b_w, a_col);
        #else
        product = dot_product_v(b_w, a_col);
        #endif
      } else if (data_rep == SPARSE32) {
        a_col_sparse = get_column(b_a_sparse, i);
        #if SCALAR
        product = dot_product(b_w, a_col_sparse);
        #else
        product = dot_product_v(b_w, a_col_sparse);
        #endif
      } else if (data_rep == QUANTIZED) {
        #if HAS_QUANTIZED
        a_col_quant = get_column(b_a_quant, i);
        a_col_quant.data->restore(b_acol32[0]);
        #if SCALAR
        product = dot_product(b_w,
            {b_acol32[0].getData(), a_col_quant.length});
        #else
        product = dot_product_v(b_w,
            {b_acol32[0].getData(), a_col_quant.length});
        #endif
        #endif
      }
      tau = ln / norm;
      gamma = (old_alpha * norm - product) / norm;
      sign = (gamma == 0.0) ? 0.0 : (gamma > 0.0 ? 1.0 : -1.0);
      new_alpha = sign * std::max((real)0.0, (std::abs(gamma) - tau));
      set_value(b_alpha, i, new_alpha);
      if (data_rep == DENSE32) {
        #if SCALAR
        scalar_multiply_add(b_w, a_col, new_alpha - old_alpha);
        #else
        scalar_multiply_add_v(b_w, a_col, new_alpha - old_alpha);
        #endif
      } else if (data_rep == SPARSE32) {
        #if SCALAR
        scalar_multiply_add(b_w, a_col_sparse, new_alpha - old_alpha);
        #else
        scalar_multiply_add_v(b_w, a_col_sparse, new_alpha - old_alpha);
        #endif
      } else if (data_rep == QUANTIZED) {
        #if HAS_QUANTIZED
        #if SCALAR
        scalar_multiply_add(b_w,
            {b_acol32[0].getData(), a_col_quant.length},
            new_alpha - old_alpha);
        #else
        scalar_multiply_add_v(b_w,
            {b_acol32[0].getData(), a_col_quant.length},
            new_alpha - old_alpha);
        #endif
        #endif
      }
    }
  }
}

void TaskB::run_svm_sequential() {
  shuffle();
  real ld = l * d;
  real label, product, norm, delta, old_alpha, new_alpha;
  real gap1, gap2;
  SparseVector a_col_sparse;
  QuantVector a_col_quant;
  Vector a_col;
  uint32_t i;
  for (uint32_t s = 0; s < p; ++s) {
    i = s;
    norm = get_value(b_norms, i);
    if (norm == 0) {
      set_value(b_alpha, i, 0);
    } else {
      label = get_value(b_b, i);
      old_alpha = get_value(b_alpha, i);
      if (data_rep == DENSE32) {
        a_col = get_column(b_a, i);
        #if SCALAR
        product = dot_product(b_w, a_col);
        #else
        product = dot_product_v(b_w, a_col);
        #endif
      } else if (data_rep == SPARSE32) {
        a_col_sparse = get_column(b_a_sparse, i);
        #if SCALAR
        product = dot_product(b_w, a_col_sparse);
        #else
        product = dot_product_v(b_w, a_col_sparse);
        #endif
      } else if (data_rep == QUANTIZED) {
        #if HAS_QUANTIZED
        a_col_quant = get_column(b_a_quant, i);
        a_col_quant.data->restore(b_acol32[0]);
        #if SCALAR
        product = dot_product(b_w,
            {b_acol32[0].getData(), a_col_quant.length});
        #else
        product = dot_product_v(b_w,
            {b_acol32[0].getData(), a_col_quant.length});
        #endif
        #endif
      }
      delta = (label - product / ld) / (norm / ld);
      new_alpha = label * std::max((real)0.0, std::min((real)1.0,
          label * (old_alpha + delta)));
      set_value(b_alpha, i, new_alpha);
      if (data_rep == DENSE32) {
        #if SCALAR
        scalar_multiply_add(b_w, a_col, new_alpha - old_alpha);
        #else
        scalar_multiply_add_v(b_w, a_col, new_alpha - old_alpha);
        #endif
      } else if (data_rep == SPARSE32) {
        #if SCALAR
        scalar_multiply_add(b_w, a_col_sparse, new_alpha - old_alpha);
        #else
        scalar_multiply_add_v(b_w, a_col_sparse, new_alpha - old_alpha);
        #endif
      } else if (data_rep == QUANTIZED) {
        #if HAS_QUANTIZED
        #if SCALAR
        scalar_multiply_add(b_w,
            {b_acol32[0].getData(), a_col_quant.length},
            new_alpha - old_alpha);
        #else
        scalar_multiply_add_v(b_w,
            {b_acol32[0].getData(), a_col_quant.length},
            new_alpha - old_alpha);
        #endif
        #endif
      }
    }
  }
}
