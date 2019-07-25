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

#include "vectorized.h"

#if !SCALAR


#if REPLACE_INTRINSICS
float _mm512_reduce_add_ps(__m512 v) {
  float out[16];
  __m512 v1 = _mm512_permute_ps(v, 0xb1);
  __m512 s1 = _mm512_add_ps(v, v1);
  __m512 v2 = _mm512_permute_ps(s1, 0x4e);
  __m512 s2 = _mm512_add_ps(s1, v2);
  __m512 v3 = _mm512_shuffle_f32x4(s2, s2, 0xb1);
  __m512 s3 = _mm512_add_ps(s2, v3);
  __m512 v4 = _mm512_shuffle_f32x4(s3, s3, 0x4e);
  __m512 s4 = _mm512_add_ps(s3, v4);
  _mm512_storeu_ps(out, s4);
  return out[0];
}

__m512 _mm512_abs_ps(__m512 v) {
  __m512 mone = _mm512_set1_ps(-1);
  __m512 neg = _mm512_mul_ps(v, mone);
  return _mm512_max_ps(v, neg);
}
#endif

real dot_product_v(QuantVector vec1, QuantVector vec2,
    uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  return vec1.data->dot(*(vec2.data), start, end);
  #else
  return 0;
  #endif
}

real dot_product_v(Vector vec1, QuantVector vec2,
    uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  return vec2.data->dot(vec1.data, start, end);
  #else
  return 0;
  #endif
}

void scalar_multiply_add_v(QuantVector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  target.data->scaleAndAdd(*(vec.data), scalar, start, end);
  #endif
}

void scalar_multiply_add_v(Vector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  vec.data->scaleAndAddOther(target.data, scalar, start, end);
  #endif
}

real dot_product_v(QuantVector vec1, QuantVector vec2) {
  #if HAS_QUANTIZED
  return dot_product_v(vec1, vec2, 0, vec1.length);
  #else
  return 0;
  #endif
}

real dot_product_v(Vector vec1, QuantVector vec2) {
  #if HAS_QUANTIZED
  return dot_product_v(vec1, vec2, 0, vec1.length);
  #else
  return 0;
  #endif
}

void scalar_multiply_add_v(QuantVector target, QuantVector vec,
    real scalar) {
  #if HAS_QUANTIZED
  scalar_multiply_add_v(target, vec, scalar, 0, vec.length);
  #endif
}

real norm_2_squared_v(QuantVector vec) {
  #if HAS_QUANTIZED
  return vec.data->dot(*(vec.data));
  #else
  return 0;
  #endif
}

void scalar_multiply_v(QuantVector target, Vector vec, real scalar) {
  #if HAS_QUANTIZED
  CloverVector32 vec32(target.length);
  Vector temp = {vec32.getData(), target.length};
  scalar_multiply_v(temp, vec, scalar);
  target.data->quantize(vec32);
  #endif
}

void scalar_divide_v(QuantVector target, Vector vec, real scalar) {
  #if HAS_QUANTIZED
  CloverVector32 vec32(target.length);
  Vector temp = {vec32.getData(), target.length};
  scalar_divide_v(temp, vec, scalar);
  target.data->quantize(vec32);
  #endif
}

void scalar_multiply_ps(Vector target, Vector vec, float scalar) {
  if (target.length != vec.length)
    raise_error("scalar_multiply: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  __m512 y = _mm512_set1_ps(scalar);
  float *v = (float*)vec.data;
  float *t = (float*)target.data;
  uint32_t i;
  uint32_t end_div_avx = target.length - (target.length & 15); 
  for (i = 0; i < end_div_avx; i += 16)
    _mm512_store_ps(t + i, _mm512_mul_ps(_mm512_load_ps(v + i), y));
  for (i = end_div_avx; i < target.length; ++i)
    t[i] = v[i] * scalar;
}

void scalar_divide_ps(Vector target, Vector vec, float scalar,
    uint32_t start, uint32_t end) {
  if (target.length != vec.length)
    raise_error("scalar_divide: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  __m512 y = _mm512_set1_ps(scalar);
  float *v = (float*)vec.data;
  float *t = (float*)target.data;
  uint32_t i;
  uint32_t end_div_avx = end - ((end - start) & 15); 
  for (i = start; i < end_div_avx; i += 16)
    _mm512_store_ps(t + i, _mm512_div_ps(_mm512_load_ps(v + i), y));
  for (i = end_div_avx; i < end; ++i)
    t[i] = v[i] / scalar;
}

float dot_product_ps(Vector vec1, Vector vec2,
    uint32_t start, uint32_t end) {
  if (vec1.length != vec2.length)
    raise_error("dot_product: Vector lengths do not match! ("
        + std::to_string(vec1.length) + "!="
        + std::to_string(vec2.length) + ")");
  if (end > vec1.length)
    raise_error("dot_product: Range out of bounds! ("
        + std::to_string(end) + ">"
        + std::to_string(vec1.length) + ")");
  __m512 sum;
  __m512 sum1 = _mm512_setzero_ps();
  __m512 sum2 = _mm512_setzero_ps();
  __m512 sum3 = _mm512_setzero_ps();
  __m512 sum4 = _mm512_setzero_ps();
  __m512 sum5 = _mm512_setzero_ps();
  __m512 sum6 = _mm512_setzero_ps();
  float fsum = 0;
  float *vector1 = (float*)vec1.data;
  float *vector2 = (float*)vec2.data;
  const float *v1;
  const float *v2;
  uint32_t i;
  uint32_t end_div_acc_avx = end - ((end - start) % 96); //acc * 16
  uint32_t end_div_avx = end - ((end - start) & 15); // % 16
  for (i = start; i < end_div_acc_avx; i += 96) {
    v1 = vector1 + i;
    v2 = vector2 + i;
    sum1 = _mm512_fmadd_ps(_mm512_load_ps(v1),
        _mm512_load_ps(v2), sum1);
    sum2 = _mm512_fmadd_ps(_mm512_load_ps(v1 + 16),
        _mm512_load_ps(v2 + 16), sum2);
    sum3 = _mm512_fmadd_ps(_mm512_load_ps(v1 + 32),
        _mm512_load_ps(v2 + 32), sum3);
    sum4 = _mm512_fmadd_ps(_mm512_load_ps(v1 + 48),
        _mm512_load_ps(v2 + 48), sum4);
    sum5 = _mm512_fmadd_ps(_mm512_load_ps(v1 + 64),
        _mm512_load_ps(v2 + 64), sum5);
    sum6 = _mm512_fmadd_ps(_mm512_load_ps(v1 + 80),
        _mm512_load_ps(v2 + 80), sum6);
  }
  sum1 = _mm512_add_ps(sum1, sum2);
  sum3 = _mm512_add_ps(sum3, sum4);
  sum5 = _mm512_add_ps(sum5, sum6);
  sum1 = _mm512_add_ps(sum1, sum3);
  sum = _mm512_add_ps(sum1, sum5);
  for (i = end_div_acc_avx; i < end_div_avx; i += 16)
    sum = _mm512_fmadd_ps(_mm512_load_ps(vector1 + i),
        _mm512_load_ps(vector2 + i), sum);
  fsum = _mm512_reduce_add_ps(sum);
  for (i = end_div_avx; i < end; ++i)
    fsum += vector1[i] * vector2[i];
  return fsum;
}

void scalar_multiply_add_ps(Vector target, Vector vec, float scalar,
    uint32_t start, uint32_t end) {
  if (target.length != vec.length)
    raise_error("scalar_multiply_add: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  if (end > vec.length)
    raise_error("scalar_multiply_add: Range out of bounds! ("
        + std::to_string(end) + ">" + std::to_string(vec.length) + ")");
  __m512 y = _mm512_set1_ps(scalar);
  float *v = (float*)vec.data;
  float *t = (float*)target.data;
  uint32_t i;
  uint32_t end_div_avx = end - ((end - start) & 15); 
  for (i = start; i < end_div_avx; i += 16)
    _mm512_store_ps(t + i, _mm512_fmadd_ps(
        _mm512_load_ps(v + i), y, _mm512_load_ps(t + i)));
  for (i = end_div_avx; i < end; ++i)
    t[i] += v[i] * scalar;
}

float norm_2_squared_ps(Vector vec){
  return dot_product_ps(vec, vec, 0, vec.length);
}
    
float norm_1_ps(Vector vec) {
  __m512 sum;
  __m512 sum1 = _mm512_setzero_ps();
  __m512 sum2 = _mm512_setzero_ps();
  __m512 sum3 = _mm512_setzero_ps();
  __m512 sum4 = _mm512_setzero_ps();
  __m512 sum5 = _mm512_setzero_ps();
  __m512 sum6 = _mm512_setzero_ps();
  float fsum = 0;
  float *vector1 = (float*)vec.data;
  const float *v1;
  uint32_t i;
  uint32_t end_div_acc_avx = vec.length - (vec.length % 96); //acc * 16
  uint32_t end_div_avx = vec.length - (vec.length & 15); // % 16
  for (i = 0; i < end_div_acc_avx; i += 96) {
    v1 = vector1 + i;
    sum1 = _mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(v1)), sum1);
    sum2 = _mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(v1 + 16)), sum2);
    sum3 = _mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(v1 + 32)), sum3);
    sum4 = _mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(v1 + 48)), sum4);
    sum5 = _mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(v1 + 64)), sum5);
    sum6 = _mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(v1 + 80)), sum6);
  }
  sum1 = _mm512_add_ps(sum1, sum2);
  sum3 = _mm512_add_ps(sum3, sum4);
  sum5 = _mm512_add_ps(sum5, sum6);
  sum1 = _mm512_add_ps(sum1, sum3);
  sum = _mm512_add_ps(sum1, sum5);
  for (i = end_div_acc_avx; i < end_div_avx; i += 16)
    sum = _mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(vector1 + i)),
        sum);
  fsum = _mm512_reduce_add_ps(sum);
  for (i = end_div_avx; i < vec.length; ++i)
    fsum += std::abs(vector1[i]);
  return fsum;
}

void scalar_multiply_v(Vector target, Vector vec, real scalar) {
  scalar_multiply_ps(target, vec, scalar);
}

void scalar_divide_v(Vector target, Vector vec, real scalar,
    uint32_t start, uint32_t end) {
  scalar_divide_ps(target, vec, scalar, start, end);
}

void scalar_divide_v(Vector target, Vector vec, real scalar) {
  scalar_divide_v(target, vec, scalar, 0, target.length);
}

real dot_product_v(Vector vec1, Vector vec2,
    uint32_t start, uint32_t end) {
  return dot_product_ps(vec1, vec2, start, end);
}

void scalar_multiply_add_v(Vector target, Vector vec, real scalar,
    uint32_t start, uint32_t end) {
  scalar_multiply_add_ps(target, vec, scalar, start, end);
}

real dot_product_v(Vector vec1, Vector vec2) {
  return dot_product_v(vec1, vec2, 0, vec1.length);
}

void scalar_multiply_add_v(Vector target, Vector vec, real scalar) {
  scalar_multiply_add_v(target, vec, scalar, 0, target.length);
}

real norm_2_squared_v(Vector vec) {
  return norm_2_squared_ps(vec);
}

real norm_1_v(Vector vec) {
  return norm_1_ps(vec);
}

float dot_product_v(Vector vec1, SparseVector vec2,
    uint32_t start, uint32_t end) {
  if (vec1.length != vec2.length)
    raise_error("dot_product: Vector lengths do not match! ("
        + std::to_string(vec1.length) + "!="
        + std::to_string(vec2.length) + ")");
  if (end > vec1.length)
    raise_error("dot_product: Range out of bounds! ("
        + std::to_string(end) + ">"
        + std::to_string(vec1.length) + ")");
  SparsePiece* ptr = vec2.data;
  uint32_t large_ctr = vec2.nnz;
  uint32_t ith_piece = 0;
  float fsum = 0;
  
  if (vec2.max_idx >= start) {
    
    __m512 v11, v12, v21, v22;
    __m512i vindex1, vindex2;
    __m512 sum1 = _mm512_set1_ps(0);
    __m512 sum2 = _mm512_set1_ps(0);
    __m512i next_vindex;
    
    if (start == 0 && end == vec1.length) {
      while (ptr) {
        
        for (uint32_t ctr = 0; ctr < ptr->small_len; ctr += 32) {
          v21 = _mm512_load_ps(ptr->values + ctr);
          v22 = _mm512_load_ps(ptr->values + ctr + 16);
          //gather
          vindex1 = _mm512_load_epi32(ptr->indices + ctr);
          vindex2 = _mm512_load_epi32(ptr->indices + ctr + 16);
          v11 = _mm512_i32gather_ps(vindex1, vec1.data, 4);
          v12 = _mm512_i32gather_ps(vindex2, vec1.data, 4);
          //fmadd
          sum1 = _mm512_fmadd_ps(v11, v21, sum1);
          sum2 = _mm512_fmadd_ps(v12, v22, sum2);
        }
        ptr = ptr->next;
      }
      fsum += _mm512_reduce_add_ps(sum1) + _mm512_reduce_add_ps(sum2);
    } else {
      raise_error("Multiple threads per vector not supported in sparse representations!");
    }
  }
  return fsum;
}

float dot_product_v(SparseVector vec1, Vector vec2) {
  return dot_product_v(vec2, vec1, 0, vec1.length);
}

float dot_product_v(Vector vec1, SparseVector vec2) {
  return dot_product_v(vec1, vec2, 0, vec1.length);
}

float dot_product_v(SparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end) {
  return dot_product_v(vec2, vec1, start, end);
}

void scalar_multiply_add_v(Vector target, SparseVector vec, real scalar,
    uint32_t start, uint32_t end) {
  if (target.length != vec.length)
    raise_error("scalar_multiply_add: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  if (end > vec.length)
    raise_error("scalar_multiply_add: Range out of bounds! ("
        + std::to_string(end) + ">" + std::to_string(vec.length) + ")");
  SparsePiece* ptr = vec.data;
  uint32_t large_ctr = vec.nnz;
  uint32_t ith_piece = 0;
  
  __m512 v1, t1, v2, t2;
  __m512i vindex1, vindex2;
  __m512 y = _mm512_set1_ps(scalar);
  __m512i next_vindex;
      
  if (start == 0 && end == target.length) {
    while (ptr) {
      for (uint32_t ctr = 0; ctr < ptr->small_len; ctr += 32) {
        v1 = _mm512_load_ps(ptr->values + ctr);
        v2 = _mm512_load_ps(ptr->values + ctr + 16);
        //gather
        vindex1 = _mm512_load_epi32(ptr->indices + ctr);
        vindex2 = _mm512_load_epi32(ptr->indices + ctr + 16);
        t1 = _mm512_i32gather_ps(vindex1, target.data, 4);
        t2 = _mm512_i32gather_ps(vindex2, target.data, 4);
        //fmadd and scatter
        _mm512_i32scatter_ps(target.data, vindex1,
            _mm512_fmadd_ps(v1, y, t1), 4);
        _mm512_i32scatter_ps(target.data, vindex2,
            _mm512_fmadd_ps(v2, y, t2), 4);
      }
      ptr = ptr->next;
    }
  } else {
    if (vec.max_idx >= start) {

      while (ptr && ptr->next && ptr->next->indices[0] <= start) {
        ptr = ptr->next;
        ++ith_piece;
      }
      while (ptr && ptr->next) {
        uint32_t ctr = 0;
        uint32_t small_ctr = ptr->small_len;
        while (ctr < small_ctr && ptr->indices[ctr] < start)
          ++ctr;
        uint32_t down_ctr = (((ctr + 31) >> 5) << 5); //round to nearest div by 16
        while (ctr < small_ctr && ctr < down_ctr
            && ptr->indices[ctr] < end) {
          target.data[ptr->indices[ctr]] += ptr->values[ctr] * scalar;
          ++ctr;
        }
        if (ctr < small_ctr && ctr < down_ctr) return;
        while (ctr + 32 < small_ctr && ptr->indices[ctr + 32] < end) {
          v1 = _mm512_load_ps(ptr->values + ctr);
          v2 = _mm512_load_ps(ptr->values + ctr + 16);
          //gather
          vindex1 = _mm512_load_epi32(ptr->indices + ctr);
          vindex2 = _mm512_load_epi32(ptr->indices + ctr + 16);
          t1 = _mm512_i32gather_ps(vindex1, target.data, 4);
          t2 = _mm512_i32gather_ps(vindex2, target.data, 4);
          //fmadd and scatter
          _mm512_i32scatter_ps(target.data, vindex1,
              _mm512_fmadd_ps(v1, y, t1), 4);
          _mm512_i32scatter_ps(target.data, vindex2,
              _mm512_fmadd_ps(v2, y, t2), 4);
          ctr += 32;
        }
        while (ctr < small_ctr && ptr->indices[ctr] < end) {
          target.data[ptr->indices[ctr]] += ptr->values[ctr] * scalar;
          ++ctr;
        }
        if (ctr < small_ctr) return;
        ptr = ptr->next;
        ++ith_piece;
      }
      if (ptr) {
        uint32_t ctr = 0;
        uint32_t small_ctr = ptr->small_len;
        while (ctr < small_ctr && ptr->indices[ctr] < start)
          ++ctr;
        uint32_t down_ctr = (((ctr + 31) >> 5) << 5); //round to nearest div by 16
        while (ctr < small_ctr && ctr < down_ctr
            && ptr->indices[ctr] < end) {
          target.data[ptr->indices[ctr]] += ptr->values[ctr] * scalar;
          ++ctr;
        }
        if (ctr < small_ctr && ctr < down_ctr) return;
        while (ctr + 32 < small_ctr && ptr->indices[ctr + 32] < end) {
          v1 = _mm512_load_ps(ptr->values + ctr);
          v2 = _mm512_load_ps(ptr->values + ctr + 16);
          //gather
          vindex1 = _mm512_load_epi32(ptr->indices + ctr);
          vindex2 = _mm512_load_epi32(ptr->indices + ctr + 16);
          t1 = _mm512_i32gather_ps(vindex1, target.data, 4);
          t2 = _mm512_i32gather_ps(vindex2, target.data, 4);
          //fmadd and scatter
          _mm512_i32scatter_ps(target.data, vindex1,
              _mm512_fmadd_ps(v1, y, t1), 4);
          _mm512_i32scatter_ps(target.data, vindex2,
              _mm512_fmadd_ps(v2, y, t2), 4);
          ctr += 32;
        }
        while (ctr < small_ctr && ptr->indices[ctr] < end) {
          target.data[ptr->indices[ctr]] += ptr->values[ctr] * scalar;
          ++ctr;
        }
      }
    }
  }
}

void scalar_multiply_add_v(Vector target, SparseVector vec,
    real scalar) {
  scalar_multiply_add_v(target, vec, scalar, 0, target.length);
}

//PRE: sorted arr with unique elements
uint32_t find_idx_low(uint32_t* arr, uint32_t el_b, uint32_t p) {
  uint32_t left = 0;
  uint32_t right = p;
  uint32_t mid;
  uint32_t el_a;
  if (el_b < arr[0])
    return 0;
  if (el_b > arr[p - 1])
    return p;
  while (left < right) {
    mid = (left + right) / 2;
    el_a = arr[mid];
    if (mid > 0 && el_a >= el_b && arr[mid - 1] < el_b)
      return mid;
    else if (el_a < el_b)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}

void scalar_multiply_add_v(Vector target, OneSparseVector vec,
    real scalar, uint32_t start, uint32_t end) {
  if (target.length != vec.length)
    raise_error("scalar_multiply_add: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  if (end > vec.length)
    raise_error("scalar_multiply_add: Range out of bounds! ("
        + std::to_string(end) + ">" + std::to_string(vec.length) + ")");
  uint32_t large_ctr = vec.padded_nnz;
  uint32_t ith_piece = 0;
  
  __m512 v1, t1, v2, t2;
  __m512i vindex1, vindex2;
  __m512 y = _mm512_set1_ps(scalar);
  __m512i next_vindex;
      
  if (start == 0 && end == target.length) {
    //uint32_t end_32 = 
    for (uint32_t ctr = 0; ctr < large_ctr; ctr += 32) {
      v1 = _mm512_load_ps(vec.values + ctr);
      v2 = _mm512_load_ps(vec.values + ctr + 16);
      //gather
      vindex1 = _mm512_load_epi32(vec.indices + ctr);
      vindex2 = _mm512_load_epi32(vec.indices + ctr + 16);
      t1 = _mm512_i32gather_ps(vindex1, target.data, 4);
      t2 = _mm512_i32gather_ps(vindex2, target.data, 4);
      //fmadd and scatter
      _mm512_i32scatter_ps(target.data, vindex1,
          _mm512_fmadd_ps(v1, y, t1), 4);
      _mm512_i32scatter_ps(target.data, vindex2,
          _mm512_fmadd_ps(v2, y, t2), 4);
    }
  } else {
    if (vec.max_idx >= start) {
      uint32_t ctr = find_idx_low(vec.indices, start, vec.nnz);
      uint32_t down_ctr = (((ctr + 31) >> 5) << 5);
      while (ctr < large_ctr && ctr < down_ctr
          && vec.indices[ctr] < end) {
        target.data[vec.indices[ctr]] += vec.values[ctr] * scalar;
        ++ctr;
      }
      if (ctr < large_ctr && ctr < down_ctr) return;
      while (ctr + 32 < large_ctr && vec.indices[ctr + 32] < end) {
        v1 = _mm512_load_ps(vec.values + ctr);
        v2 = _mm512_load_ps(vec.values + ctr + 16);
        //gather
        vindex1 = _mm512_load_epi32(vec.indices + ctr);
        vindex2 = _mm512_load_epi32(vec.indices + ctr + 16);
        t1 = _mm512_i32gather_ps(vindex1, target.data, 4);
        t2 = _mm512_i32gather_ps(vindex2, target.data, 4);
        //fmadd and scatter
        _mm512_i32scatter_ps(target.data, vindex1,
            _mm512_fmadd_ps(v1, y, t1), 4);
        _mm512_i32scatter_ps(target.data, vindex2, 
            _mm512_fmadd_ps(v2, y, t2), 4);
        ctr += 32;
      }
      while (ctr < large_ctr && vec.indices[ctr] < end) {
        target.data[vec.indices[ctr]] += vec.values[ctr] * scalar;
        ++ctr;
      }
    }
  }
}

void scalar_multiply_add_v(Vector target, OneSparseVector vec,
    real scalar) {
  scalar_multiply_add_v(target, vec, scalar, 0, target.length);
}

float dot_product_v(Vector vec1, OneSparseVector vec2,
    uint32_t start, uint32_t end) {
  if (vec1.length != vec2.length)
    raise_error("dot_product: Vector lengths do not match! ("
        + std::to_string(vec1.length) + "!="
        + std::to_string(vec2.length) + ")");
  if (end > vec1.length)
    raise_error("dot_product: Range out of bounds! ("
        + std::to_string(end) + ">"
        + std::to_string(vec1.length) + ")");

  float fsum = 0;
  
  if (vec2.max_idx >= start) {
    
    __m512 v11, v12, v21, v22;
    __m512i vindex1, vindex2;
    __m512 sum1 = _mm512_set1_ps(0);
    __m512 sum2 = _mm512_set1_ps(0);
    __m512i next_vindex;
    
    if (start == 0 && end == vec1.length) {
      for (uint32_t ctr = 0; ctr < vec2.padded_nnz; ctr += 32) {
        v21 = _mm512_load_ps(vec2.values + ctr);
        v22 = _mm512_load_ps(vec2.values + ctr + 16);
        //gather
        vindex1 = _mm512_load_epi32(vec2.indices + ctr);
        vindex2 = _mm512_load_epi32(vec2.indices + ctr + 16);
        v11 = _mm512_i32gather_ps(vindex1, vec1.data, 4);
        v12 = _mm512_i32gather_ps(vindex2, vec1.data, 4);
        //fmadd
        sum1 = _mm512_fmadd_ps(v11, v21, sum1);
        sum2 = _mm512_fmadd_ps(v12, v22, sum2);
      }
      fsum = _mm512_reduce_add_ps(sum1) + _mm512_reduce_add_ps(sum2);
    } else {
      raise_error("Multiple threads per vector not supported for sparse representations!");
    }
  }
  return fsum;
}

float dot_product_v(OneSparseVector vec1, Vector vec2) {
  return dot_product_v(vec2, vec1, 0, vec1.length);
}

float dot_product_v(Vector vec1, OneSparseVector vec2) {
  return dot_product_v(vec1, vec2, 0, vec1.length);
}

float dot_product_v(OneSparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end) {
  return dot_product_v(vec2, vec1, start, end);
}

float norm_2_squared_v(OneSparseVector vec) {
  __m512 v1, v2;
  __m512 sum1 = _mm512_set1_ps(0);
  __m512 sum2 = _mm512_set1_ps(0);
  float fsum = 0;
  for (uint32_t ctr = 0; ctr < vec.padded_nnz; ctr += 32) {
    v1 = _mm512_load_ps(vec.values + ctr);
    v2 = _mm512_load_ps(vec.values + ctr + 16);
    sum1 = _mm512_fmadd_ps(v1, v1, sum1);
    sum2 = _mm512_fmadd_ps(v2, v2, sum2);
  }
  fsum = _mm512_reduce_add_ps(sum1) + _mm512_reduce_add_ps(sum2);
  return fsum;
}

float norm_2_squared_v(SparseVector vec) {
  SparsePiece* ptr = vec.data;
  uint32_t large_ctr = vec.nnz;
  uint32_t ith_piece = 0;
  real fsum = 0;
  __m512 sum = _mm512_set1_ps(0);
  __m512 x;
  while (ptr) {
    uint32_t ctr = 0;
    uint32_t small_ctr = ((ptr->next)
        ? SPARSE_PIECE_LENGTH
        : large_ctr - ith_piece * SPARSE_PIECE_LENGTH);
    uint32_t end_d16 = (small_ctr >> 4);
    for (uint32_t ctr = 0; ctr < end_d16; ctr += 16) {
      x = _mm512_load_ps(ptr->values + ctr);
      sum = _mm512_fmadd_ps(x, x, sum);
    }
    for (uint32_t ctr = end_d16; ctr < small_ctr; ++ctr)
      fsum += ptr->values[ctr] * ptr->values[ctr];
    ptr = ptr->next;
    ++ith_piece;
  }
  return fsum + _mm512_reduce_add_ps(sum);
}

#endif
