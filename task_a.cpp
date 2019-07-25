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

class TaskA::CompareIdxDesc {
private:
  Vector z;
public:
  CompareIdxDesc(Vector z) : z(z){}
  bool operator()(uint32_t idx_a, uint32_t idx_b) {
    return z.data[idx_b] < z.data[idx_a]; //bypass range checks
  }
};

inline void TaskA::initialize_vecs_common(Vector b, bool primal) {
  uint32_t rows = primal ? d : n;
  uint32_t columns = primal ? n : d;
  a_b = b;
  a_alpha = create_vector(columns, A_USE_HBW);
  a_z = create_vector(columns, A_USE_HBW);
  a_w = create_vector(rows, A_USE_HBW);
  a_norms = create_vector(columns, A_USE_HBW);
  a_p = (uint32_t*)b_malloc(columns * sizeof(uint32_t), A_USE_HBW);
  exe_order = (uint32_t*)b_malloc(columns * sizeof(uint32_t),
      A_USE_HBW);
  a_z_copy = (real*)b_malloc(columns * sizeof(real), A_USE_HBW);
  for (uint32_t i = 0; i < columns; ++i) {
    a_p[i] = i;
    exe_order[i] = i;
    set_value(a_alpha, i, 0);
  }
  if (primal) {
    #if SCALAR
    scalar_multiply(a_w, a_b, -1.0);
    bound = norm_2_squared(a_b) / (2.0 * l * d);
    #else
    scalar_multiply_v(a_w, a_b, -1.0);
    bound = norm_2_squared_v(a_b) / (2.0 * l * d);
    #endif
  }
}

inline void TaskA::initialize_vecs(OneSparseMatrix a, Vector b,
    bool primal) {
  initialize_vecs_common(b, primal);
  uint32_t rows = primal ? d : n;
  uint32_t columns = primal ? n : d;
  a_a_one_sparse = a;
  #if SCALAR
  for (uint32_t i = 0; i < columns; ++i)
    set_value(a_norms, i, norm_2_squared(get_column(a, i)));
  #else
  for (uint32_t i = 0; i < columns; ++i)
    set_value(a_norms, i, norm_2_squared_v(get_column(a, i)));
  #endif
}

inline void TaskA::initialize_vecs(QuantMatrix a, Vector b,
    bool primal) {
  initialize_vecs_common(b, primal);
  uint32_t rows = primal ? d : n;
  uint32_t columns = primal ? n : d;
  a_a_quant = a;
  #if SCALAR
  for (uint32_t i = 0; i < columns; ++i)
    set_value(a_norms, i, norm_2_squared(get_column(a, i)));
  #else
  for (uint32_t i = 0; i < columns; ++i)
    set_value(a_norms, i, norm_2_squared_v(get_column(a, i)));
  #endif
  if (primal)
    inner_transpose(a_w);
}

inline void TaskA::initialize_vecs(Matrix a, Vector b, bool primal) {
  initialize_vecs_common(b, primal);
  uint32_t rows = primal ? d : n;
  uint32_t columns = primal ? n : d;
  a_a = a;
  #if SCALAR
  for (uint32_t i = 0; i < columns; ++i)
    set_value(a_norms, i, norm_2_squared(get_column(a, i)));
  #else
  for (uint32_t i = 0; i < columns; ++i)
    set_value(a_norms, i, norm_2_squared_v(get_column(a, i)));
  #endif
}

inline void TaskA::initialize_thread_data(bool primal) {
  gap_data.a_alpha = a_alpha;
  gap_data.a_w = a_w;
  gap_data.a_b = a_b;
  gap_data.a_a = a_a;
  gap_data.a_a_one_sparse = a_a_one_sparse;
  gap_data.a_a_quant = a_a_quant;
  gap_data.a_z = a_z;
  gap_data.a_norms = a_norms;
  gap_data.bound = bound;
  gap_data.regularization = l * d;
  gap_data.ln = l * n;
  gap_data.par_updates = par_updates;
  gap_data.n_gaps = (primal ? n : d);
  gap_data.running = a_running;
  gap_data.exe_order = exe_order;
  gap_data.updated = updated_ctr;
  thread_data = (struct threaded::GapThreadArguments*)
      b_malloc(par_updates
          * sizeof(struct threaded::GapThreadArguments), A_USE_HBW);
  #if HAS_QUANTIZED
  gap_data.a_acol32 = a_acol32;
  #endif
  gap_data.cond_mutex = &a_cond_mutex;
  gap_data.cond = &a_cond;
  gap_data.bar_mutex = &a_bar_mutex;
  gap_data.bar1 = &a_bar1;
  gap_data.bar2 = &a_bar2;
  gap_data.can_start = &a_can_start;
  gap_data.threads_running = &a_threads_running;
  gap_data.primal = primal;
  gap_data.random_engine = &random_engine;
  gap_data.b_only = b_only;
}

inline void TaskA::initialize_common_pre(
    uint32_t samples, uint32_t features, uint32_t b_elements,
    real lambda, uint32_t a_par_updates, uint32_t b_par_updates,
    uint32_t b_threads_per_vec, bool *running, bool primal, bool b_o) {
  random_engine = std::minstd_rand0(
      std::chrono::system_clock::now().time_since_epoch().count());
  l = lambda;
  d = samples;
  n = features;
  p = b_elements;
  duality_gap = 1;
  a_running = running;
  par_updates = a_par_updates;
  a_threads = (pthread_t*)b_malloc(par_updates * sizeof(pthread_t),
      A_USE_HBW);
  updated_ctr = (uint32_t*)b_malloc(par_updates * sizeof(uint32_t),
      A_USE_HBW);
  b_thread_offset = (b_o ? 0 : b_threads_per_vec * b_par_updates);
  b_only = b_o;
}

inline void TaskA::initialize_common_post(bool primal) {
  initialize_thread_data(primal);
  threaded::initialize_a(par_updates, primal ? n : d, A_USE_HBW);
  init_threads();
}

void TaskA::initialize(OneSparseMatrix a, Vector b,
    uint32_t samples, uint32_t features, uint32_t b_elements,
    real lambda, uint32_t a_par_updates, uint32_t b_par_updates,
    uint32_t b_threads_per_vec, bool *running, bool primal,
    bool b_only) {
  initialize_common_pre(samples, features, b_elements, lambda,
      a_par_updates, b_par_updates, b_threads_per_vec, running, primal,
      b_only);
  initialize_vecs(a, b, primal);
  initialize_common_post(primal);
}

void TaskA::initialize(QuantMatrix a, Vector b,
    uint32_t samples, uint32_t features, uint32_t b_elements,
    real lambda, uint32_t a_par_updates, uint32_t b_par_updates,
    uint32_t b_threads_per_vec, bool *running, bool primal,
    bool b_only) {
  initialize_common_pre(samples, features, b_elements, lambda,
      a_par_updates, b_par_updates, b_threads_per_vec, running, primal,
      b_only);
  #if HAS_QUANTIZED
  a_acol32 = (CloverVector32*)b_malloc(
      par_updates * sizeof(CloverVector32), A_USE_HBW);
  for (uint32_t i = 0; i < par_updates; ++i)
    new(a_acol32 + i) CloverVector32(primal ? d : n);
  #endif
  initialize_vecs(a, b, primal);
  initialize_common_post(primal);
}

void TaskA::initialize(Matrix a, Vector b,
    uint32_t samples, uint32_t features, uint32_t b_elements,
    real lambda, uint32_t a_par_updates, uint32_t b_par_updates,
    uint32_t b_threads_per_vec, bool *running, bool primal,
    bool b_only) {
  initialize_common_pre(samples, features, b_elements, lambda,
      a_par_updates, b_par_updates, b_threads_per_vec, running, primal,
      b_only);
  initialize_vecs(a, b, primal);
  initialize_common_post(primal);
}

void TaskA::deinitialize() {
  join_threads();
  destroy(a_alpha, A_USE_HBW);
  destroy(a_norms, A_USE_HBW);
  destroy(a_z, A_USE_HBW);
  destroy(a_w, A_USE_HBW);
  #if HAS_QUANTIZED
  if (data_rep == QUANTIZED) {
    for (uint32_t i = 0; i < par_updates; ++i)
      a_acol32[i].~CloverVector32();
    b_free(a_acol32, A_USE_HBW);
  }
  #endif
  b_free(a_p, A_USE_HBW);
  b_free(a_threads, A_USE_HBW);
  b_free(exe_order, A_USE_HBW);
  b_free(thread_data, A_USE_HBW);
  b_free(updated_ctr, A_USE_HBW);
  b_free(a_z_copy, A_USE_HBW);
  threaded::deinitialize_a(par_updates, A_USE_HBW);
}

inline real TaskA::partition(uint32_t size) {
  std::memcpy(a_z_copy, a_z.data, size * sizeof(real));
  std::nth_element(a_z_copy, a_z_copy + size - p, a_z_copy + size);
  return a_z_copy[size - p];
}

void TaskA::update_p(bool primal) {
  uint32_t size = primal ? n : d;
  uint32_t j = 0;
  real partition_val = partition(size);
  for (uint32_t i = 0; i < size && j < p; ++i) {
    if (get_value(a_z, i) > partition_val) {
      a_p[j] = i;
      ++j;
    }
  }
  if (j < p) {
    for (uint32_t i = 0; i < size && j < p; ++i) {
      if (get_value(a_z, i) == partition_val) {
        a_p[j] = i;
        ++j;
      }
    }
  }
  if (j < p) {
    for (uint32_t i = 0; i < size && j < p; ++i) {
      if (get_value(a_z, i) < partition_val) {
        a_p[j] = i;
        ++j;
      }
    }
  }
}

void a_nsleep() {
  struct timespec tim;
  tim.tv_sec = 0;
  tim.tv_nsec = 0;
  nanosleep(&tim, NULL);
}

void* update_gaps(void* data) {
  struct threaded::GapArguments *args
      = ((struct threaded::GapThreadArguments*)data)->args;
  while (*(args->threads_running)) {
    pthread_mutex_lock(args->cond_mutex);
    if (!*(args->can_start))
      pthread_cond_wait(args->cond, args->cond_mutex);
    pthread_mutex_unlock(args->cond_mutex);
    if (*(args->threads_running)) {
      
      if (data_rep == DENSE32) {
        if(args->primal)
          threaded::update_z_lasso_dense(data);
        else
          threaded::update_z_svm_dense(data);
      } else if (data_rep == SPARSE32) {
        if(args->primal)
          threaded::update_z_lasso_sparse(data);
        else
          threaded::update_z_svm_sparse(data);
      } else if (data_rep == QUANTIZED) {
        if(args->primal)
          threaded::update_z_lasso_quantized(data);
        else
          threaded::update_z_svm_quantized(data);
      }

      pthread_mutex_lock(args->bar_mutex);
      (*(args->bar1))++;
      pthread_mutex_unlock(args->bar_mutex);
      while (*(args->bar1) != 0) a_nsleep();
      pthread_mutex_lock(args->bar_mutex);
      (*(args->bar2))++;
      pthread_mutex_unlock(args->bar_mutex);
      while (*(args->bar2) != 0) a_nsleep();
    }
  }
  pthread_exit(0);
}

void TaskA::init_threads() {
  a_can_start = false;
  pthread_cond_init(&a_cond, NULL);
  pthread_mutex_init(&a_cond_mutex, NULL);
  pthread_mutex_init(&a_bar_mutex, NULL);
  a_threads_running = true;
  a_can_start = false;
  pthread_attr_init(&attr);
  cpu_set_t cpuset;
  for (uint32_t t = 0; t < par_updates; ++t) {
    thread_data[t].update_thread_id = t;
    thread_data[t].args = &gap_data;
    uint32_t affinity = t + TILE_MULTIPLIER * b_thread_offset
        + CORE_OFFSET;
    CPU_ZERO(&cpuset);
    CPU_SET(affinity, &cpuset);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
    pthread_create(&a_threads[t], &attr, update_gaps,
        (void*)&thread_data[t]);
  }
  a_bar1 = 0;
  a_bar2 = 0;
}

void TaskA::join_threads() {
  a_threads_running = false;
  tStart();
  for (uint32_t t = 0; t < par_updates; ++t)
    pthread_join(a_threads[t], NULL);
  pthread_attr_destroy(&attr);
  pthread_cond_destroy(&a_cond);
  pthread_mutex_destroy(&a_cond_mutex);
  pthread_mutex_destroy(&a_bar_mutex);
}

void TaskA::tStart() {
  pthread_mutex_lock(&a_cond_mutex);
  pthread_cond_broadcast(&a_cond);
  a_can_start = true;
  pthread_mutex_unlock(&a_cond_mutex);
}

void TaskA::tStop() {
  a_can_start = false;
}

void TaskA::run_lasso() {
  std::memset(updated_ctr, 0, par_updates * sizeof(uint32_t));
  tStart();
  while (a_bar1 < par_updates) a_nsleep();
  a_bar1 = 0;
  tStop();
  while (a_bar2 < par_updates) a_nsleep();
  a_bar2 = 0;
}

inline void TaskA::update_z_lasso(uint32_t index) {
  if (get_value(a_norms, index) == 0) {
    set_value(a_z, index, 0);
  } else {
    real alpha = get_value(a_alpha, index);
    real ld = l * d;
    real product;
    if (data_rep == DENSE32) {
      #if SCALAR
      product = dot_product(a_w, get_column(a_a, index));
      #else
      product = dot_product_v(a_w, get_column(a_a, index));
      #endif
    } else if (data_rep == SPARSE32) {
      #if SCALAR
      product = dot_product(a_w, get_column(a_a_one_sparse, index));
      #else
      product = dot_product_v(a_w, get_column(a_a_one_sparse, index));
      #endif
    } else if (data_rep == QUANTIZED) {
      #if HAS_QUANTIZED
      get_column(a_a_quant, index).data->restore(a_acol32[0]);
      #if SCALAR
      product = dot_product(a_w, {a_acol32[0].getData(), a_w.length});
      #else
      product = dot_product_v(a_w, {a_acol32[0].getData(), a_w.length});
      #endif
      #endif
    }
    real new_z = alpha * product + ld * std::abs(alpha)
        + bound * std::max((real)0.0, std::abs(product) - ld);
    set_value(a_z, index, new_z);
  }
}

void TaskA::run_lasso_sequential() {
  for(uint32_t s = 0; s < n; ++s)
    update_z_lasso(exe_order[s]);
}

void TaskA::update_lasso_stats() {
  real gap_sum = 0;
  for (uint32_t i = 0; i < n; ++i)
    gap_sum += get_value(a_z, i);
  #if SCALAR
  objective = norm_2_squared(a_w) / (2 * d) + l * norm_1(a_alpha);
  #else
  objective = norm_2_squared_v(a_w) / (2 * d) + l * norm_1_v(a_alpha);
  #endif
  duality_gap = gap_sum / d;
  total_updated = 0;
  for (uint32_t t = 0; t < par_updates; ++t)
    total_updated += updated_ctr[t];
}

void TaskA::run_svm() {
  std::memset(updated_ctr, 0, par_updates * sizeof(uint32_t));
  tStart();
  while (a_bar1 < par_updates) a_nsleep();
  a_bar1 = 0;
  tStop();
  while (a_bar2 < par_updates) a_nsleep();
  a_bar2 = 0;
}

inline void TaskA::update_z_svm(uint32_t index) {
  if (get_value(a_norms, index) == 0) {
    set_value(a_z, index, 0);
  } else {
    real alpha = get_value(a_alpha, index);
    real score;
    if (data_rep == DENSE32) {
      #if SCALAR
      score = dot_product(get_column(a_a, index), a_w) / (l * d);
      #else
      score = dot_product_v(get_column(a_a, index), a_w) / (l * d);
      #endif
    } else if (data_rep == SPARSE32) {
      #if SCALAR
      score = dot_product(
          get_column(a_a_one_sparse, index), a_w) / (l * d);
      #else
      score = dot_product_v(
          get_column(a_a_one_sparse, index), a_w) / (l * d);
      #endif
    } else if (data_rep == QUANTIZED) {
      #if HAS_QUANTIZED
      get_column(a_a_quant, index).data->restore(a_acol32[0]);
      #if SCALAR
      score = dot_product(
          a_w, {a_acol32[0].getData(), a_w.length}) / (l * d);
      #else
      score = dot_product_v(
          a_w, {a_acol32[0].getData(), a_w.length}) / (l * d);
      #endif
      #endif
    }
    real label = get_value(a_b, index);
    real new_z = score * alpha
        + std::max((real)0.0, (real)1.0 - label * score)
        - label * alpha;
    set_value(a_z, index, new_z);
  }
}

void TaskA::run_svm_sequential() {
  for (uint32_t s = 0; s < d; ++s)
    update_z_svm(exe_order[s]);
}

void TaskA::update_svm_stats() {
  real gap_sum = 0;
  for (uint32_t i = 0; i < d; ++i)
    gap_sum += get_value(a_z, i);
  #if SCALAR
  objective = norm_2_squared(a_w) / (2 * l * d * d)
      - dot_product(a_alpha, a_b) / d;
  #else
  objective = norm_2_squared_v(a_w) / (2 * l * d * d)
      - dot_product_v(a_alpha, a_b) / d;
  #endif
  duality_gap = gap_sum / d;
  total_updated = 0;
  for (uint32_t t = 0; t < par_updates; ++t)
    total_updated += updated_ctr[t];
}

void TaskA::update_alpha(Vector alpha, uint32_t *element_idx,
    uint32_t max_idx) {
  for (uint32_t i = 0; i < max_idx; ++i)
    set_value(a_alpha, element_idx[i], get_value(alpha, i));
}

void TaskA::update_w(Vector w) {
  set(a_w, w);
}

void TaskA::update_parameters(Vector alpha, Vector w) {
  set(a_alpha, alpha);
  set(a_w, w);
}

Vector TaskA::get_weights(bool primal) {
  return primal ? a_alpha : a_w;
}

real TaskA::get_duality_gap() {
  return duality_gap;
}

real TaskA::get_objective() {
  return objective;
}

real TaskA::get_total_updated() {
  return total_updated;
}

//These functions are left here for comparison against the OMP baseline
//The comparisons work only for dense data

void TaskA::shuffle(bool primal) {
  uint32_t size = primal ? n : d;
  std::shuffle(&exe_order[0], &exe_order[size], random_engine);
}

void TaskA::run_lasso_omp() {
  #pragma omp parallel for num_threads(par_updates)
  for (uint32_t s = 0; s < n; ++s) {
    if (*a_running) {
      uint32_t index = exe_order[s];
      real ld = l * d;
      real product = 0;
      real* a_col = get_column(a_a, index).data;
      real* w = a_w.data;
      #pragma omp simd reduction(+ : product)
      for(uint32_t i = 0; i < d; ++i)
        product += a_col[i] * w[i];
      real alpha = get_value(a_alpha, index);
      real new_z = alpha * product + ld * std::abs(alpha)
        + bound * std::max((real)0.0, std::abs(product) - ld);
      set_value(a_z, index, new_z);
    }
  }
}

void TaskA::run_svm_omp() {
  #pragma omp parallel for num_threads(par_updates)
  for (uint32_t s = 0; s < d; ++s) {
    if (*a_running) {
      uint32_t index = exe_order[s];
      real alpha = get_value(a_alpha, index);
      if (get_value(a_norms, index) == 0) {
          set_value(a_z, index, 0);
      } else {
        real ld = l * d;
        real product = 0;
        real* a_col = get_column(a_a, index).data;
        real* w = a_w.data;
        #pragma omp simd reduction(+ : product)
        for (uint32_t i = 0; i < n; ++i)
          product += a_col[i] * w[i];
        real score = product / ld;
        real label = get_value(a_b, index);
        real new_z = score * alpha
            + std::max((real)0.0, (real)1.0 - label * score)
            - label * alpha;
        set_value(a_z, index, new_z);
      }
    }
  }
}
