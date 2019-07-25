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

#ifndef VECTORIZED_H
#define VECTORIZED_H

#include <inttypes.h>
#include "immintrin.h"
#include "algebra.h"

#if !SCALAR

real norm_2_squared_v(Vector vec);
real norm_1_v(Vector vec);
real dot_product_v(Vector vec1, Vector vec2);
void scalar_multiply_v(Vector target, Vector vec, real scalar);
void scalar_divide_v(Vector target, Vector vec, real scalar,
    uint32_t start, uint32_t end);
void scalar_divide_v(Vector target, Vector vec, real scalar);
void scalar_multiply_add_v(Vector target, Vector vec, real scalar);
real dot_product_v(Vector vec1, Vector vec2,
    uint32_t start, uint32_t end);
void scalar_multiply_add_v(Vector target, Vector vec, real scalar,
    uint32_t start, uint32_t end);
    
real dot_product_v(QuantVector vec1, QuantVector vec2,
    uint32_t start, uint32_t end);
real dot_product_v(Vector vec1, QuantVector vec2,
    uint32_t start, uint32_t end);
void scalar_multiply_add_v(QuantVector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end);
void scalar_multiply_add_v(Vector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end);
real dot_product_v(Vector vec1, QuantVector vec2);
real dot_product_v(QuantVector vec1, QuantVector vec2);
void scalar_multiply_add_v(QuantVector target, QuantVector vec,
    real scalar);
void scalar_multiply_add_v(Vector target, QuantVector vec, real scalar);
real norm_2_squared_v(QuantVector vec);
void scalar_multiply_v(QuantVector target, Vector vec, real scalar);
void scalar_divide_v(QuantVector target, Vector vec, real scalar);

void scalar_multiply_add_v(Vector target, SparseVector vec,
    real scalar);
void scalar_multiply_add_v(Vector target, SparseVector vec,
    real scalar, uint32_t start, uint32_t end);
float dot_product_v(SparseVector vec1, Vector vec2);
float dot_product_v(Vector vec1, SparseVector vec2);
float dot_product_v(SparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end);
float dot_product_v(Vector vec1, SparseVector vec2,
    uint32_t start, uint32_t end);
float norm_2_squared_v(SparseVector vec);

void scalar_multiply_add_v(Vector target, OneSparseVector vec,
    real scalar);
void scalar_multiply_add_v(Vector target, OneSparseVector vec,
    real scalar, uint32_t start, uint32_t end);
float dot_product_v(OneSparseVector vec1, Vector vec2);
float dot_product_v(Vector vec1, OneSparseVector vec2);
float dot_product_v(OneSparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end);
float dot_product_v(Vector vec1, OneSparseVector vec2,
    uint32_t start, uint32_t end);
float norm_2_squared_v(OneSparseVector vec);

#endif

#endif
