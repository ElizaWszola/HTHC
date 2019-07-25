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

#ifndef ALGEBRA_H
#define ALGEBRA_H

//PREPROCESSOR DIRECTIVES DEFINED IN CMakeLists.txt
// SCALAR        //if true,  no explicit AVX-512 vectorization
// HAS_HBW       //if false, no high-bandwidth memory is used
// LOCK          //if false, no locks are used ("hogwild"-style)
// HAS_QUANTIZED //if false, quantized type is omitted in compilation

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cmath>
#include <chrono>
#include <random>

#if HAS_HBW
#include <hbwmalloc.h>
#endif

#if HAS_QUANTIZED
#include <CloverVector4.h>
#include <CloverVector32.h>
#endif

#define real float
#define CloverVec CloverVector4
#define int_quant int8_t

//when there are too many gaps shuffling on A takes so much time that
//no gaps are updated during B's run, thus, above some threshold,
//stochastic selection is used instead
#define SHUFFLE_THRESHOLD 262144

extern uint32_t LINE_SIZE;
extern uint32_t SPARSE_PIECE_LENGTH;
extern uint32_t B_THREAD_CHUNK_SIZE;

enum data_representations {DENSE32, SPARSE32, QUANTIZED};
extern data_representations data_rep;

void nsleep(uint32_t sleep_time);

void raise_error(std::string error);

void *b_malloc(size_t size, bool use_hbw);
void b_free(void* ptr, bool use_hbw);

struct MatrixData {
  real *data;
  uint32_t columns;
  uint32_t rows;
  uint32_t padded_rows;
};

struct VectorData {
  real *data;
  uint32_t length;
};

typedef struct MatrixData Matrix;
typedef struct VectorData Vector;

struct SparsePieceData {
  real *values;
  uint32_t* indices;
  uint32_t max_idx;
  uint32_t small_len;
  struct SparsePieceData* next;
};

typedef struct SparsePieceData SparsePiece;

struct SparseVectorData {
  SparsePiece* data;
  uint32_t length;
  uint32_t nnz;
  uint32_t padded_nnz;
  uint32_t max_idx;
};

struct SparseMatrixData {
  struct SparseVectorData* column_data;
  uint32_t columns;
  uint32_t rows;
};

typedef struct SparseMatrixData SparseMatrix;
typedef struct SparseVectorData SparseVector;

struct OneSparseVectorData {
  real *values;
  uint32_t *indices;
  uint32_t length;
  uint32_t nnz;
  uint32_t padded_nnz;
  uint32_t max_idx;
};

struct OneSparseMatrixData {
  struct OneSparseVectorData* column_data;
  uint32_t columns;
  uint32_t rows;
};

typedef struct OneSparseMatrixData OneSparseMatrix;
typedef struct OneSparseVectorData OneSparseVector;

struct QuantMatrixData {
  #if HAS_QUANTIZED
  CloverVec *data;
  uint32_t columns;
  uint32_t rows;
  uint32_t padded_rows;
  #endif
};

struct QuantVectorData {
  #if HAS_QUANTIZED
  CloverVec *data;
  uint32_t length;
  #endif
};

typedef struct QuantMatrixData QuantMatrix;
typedef struct QuantVectorData QuantVector;

Matrix create_matrix(real *data, uint32_t rows, uint32_t columns);
Vector create_vector(real *data, uint32_t length);
Matrix create_matrix(real *data, uint32_t rows, uint32_t columns,
    bool use_hbw);
Vector create_vector(real *data, uint32_t length, bool use_hbw);
Matrix create_matrix(uint32_t rows, uint32_t columns);
Vector create_vector(uint32_t length);
Matrix create_matrix(uint32_t rows, uint32_t columns, bool use_hbw);
Vector create_vector(uint32_t length, bool use_hbw);

Matrix read_matrix(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw);
Vector read_vector(std::ifstream &y_stream,
    uint32_t length, bool use_hbw);

void destroy(Matrix mat);
void destroy(Vector vec);
void destroy(Matrix mat, bool use_hbw);
void destroy(Vector vec, bool use_hbw);

Vector get_column(Matrix mat, uint32_t index);
real get_value(Matrix mat, uint32_t row, uint32_t column);
real get_value(Vector vec, uint32_t index);

void set(Vector &vec, Vector tvec);
void set_column(Matrix &mat, uint32_t column, Vector vec);
void set_value(Vector &vec, uint32_t index, real value);
void set_zero(Vector &vec);

real norm_2_squared(Vector vec);
real norm_1(Vector vec);
real dot_product(Vector vec1, Vector vec2);
real dot_product(Vector vec1, Vector vec2,
    uint32_t start, uint32_t end);
void scalar_multiply(Vector target, Vector vec, real scalar);
void scalar_divide(Vector target, Vector vec, real scalar);
void scalar_multiply_add(Vector target, Vector vec, real scalar);
void scalar_multiply_add(Vector target, Vector vec, real scalar,
    uint32_t start, uint32_t end);

SparseMatrix create_matrix_sparse(uint32_t rows, uint32_t columns,
    bool use_hbw);
SparseMatrix read_matrix_sparse(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw);
void destroy(SparseMatrix mat);
void destroy(SparseMatrix mat, bool use_hbw);

SparseVector get_column(SparseMatrix mat, uint32_t index);
void set_column(SparseMatrix &mat, uint32_t column, SparseVector vec);  
real dot_product(Vector vec1, SparseVector vec2);
real dot_product(SparseVector vec1, Vector vec2);
real dot_product(Vector vec1, SparseVector vec2,
    uint32_t start, uint32_t end);
real dot_product(SparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end);
void scalar_multiply_add(Vector target, SparseVector vec, real scalar);
void scalar_multiply_add(Vector target, SparseVector vec, real scalar,
    uint32_t start, uint32_t end);
real norm_2_squared(SparseVector vec);

real dot_product(OneSparseVector vec1, Vector vec2);
real dot_product(Vector vec1, OneSparseVector vec2);
real dot_product(OneSparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end);
real dot_product(Vector vec1, OneSparseVector vec2,
    uint32_t start, uint32_t end);
void scalar_multiply_add(Vector target, OneSparseVector vec,
    real scalar);
void scalar_multiply_add(Vector target, OneSparseVector vec,
    real scalar, uint32_t start, uint32_t end);
real norm_2_squared(OneSparseVector vec);

OneSparseMatrix read_matrix_one_sparse(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw);
void destroy(OneSparseMatrix mat);
void destroy(OneSparseMatrix mat, bool use_hbw);

OneSparseVector get_column(OneSparseMatrix mat, uint32_t index);
void set_column(SparseMatrix &mat, uint32_t column,
    OneSparseVector vec);

void copy(Vector target, Vector source, uint32_t start, uint32_t end);

real dot_product(QuantVector vec1, QuantVector vec2,
    uint32_t start, uint32_t end);
real dot_product(Vector vec1, QuantVector vec2,
    uint32_t start, uint32_t end);
void scalar_multiply_add(QuantVector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end);
void scalar_multiply_add(Vector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end);
real dot_product(QuantVector vec1, QuantVector vec2);
real dot_product(Vector vec1, QuantVector vec2);
void scalar_multiply_add(QuantVector target, QuantVector vec,
    real scalar);
real norm_2_squared(QuantVector vec);
void scalar_multiply(QuantVector target, Vector vec, real scalar);
void scalar_divide(QuantVector target, Vector vec, real scalar);

QuantMatrix create_matrix_quant(real *data,
    uint32_t rows, uint32_t columns);
QuantVector create_vector_quant(real *data, uint32_t length);
QuantMatrix create_matrix_quant(real *data,
    uint32_t rows, uint32_t columns, bool use_hbw);
QuantVector create_vector_quant(real *data, uint32_t length,
    bool use_hbw);
QuantMatrix create_matrix_quant(uint32_t rows, uint32_t columns);
QuantVector create_vector_quant(uint32_t length);
QuantMatrix create_matrix_quant(uint32_t rows, uint32_t columns,
    bool use_hbw);
QuantVector create_vector_quant(uint32_t length, bool use_hbw);

QuantMatrix read_matrix_quant(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw);
QuantVector read_vector_quant(std::ifstream &y_stream,
    uint32_t length, bool use_hbw);

void destroy(QuantMatrix mat);
void destroy(QuantVector vec);
void destroy(QuantMatrix mat, bool use_hbw);
void destroy(QuantVector vec, bool use_hbw);

QuantVector get_column(QuantMatrix mat, uint32_t index);
void set_column(QuantMatrix &mat, uint32_t column, QuantVector vec);
void set_zero(QuantVector &vec);

void copy(QuantVector target, QuantVector source,
    uint32_t start, uint32_t end);

void inner_transpose(Vector vec);

    
#endif
