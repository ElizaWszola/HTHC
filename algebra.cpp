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

#include "algebra.h"

//The non-vectorized functions are not really optimized.
//They are here to validate correctness.

//The quantized functions do nothing if HAS_QUANTIZED is disabled
//With HAS_QUANTIZED disabled and main.cpp implemented the way it is
//it should not be possible to call them anyway

uint32_t LINE_SIZE;
uint32_t SPARSE_PIECE_LENGTH;
uint32_t B_THREAD_CHUNK_SIZE;
data_representations data_rep;

void nsleep(uint32_t sleep_time) {
  struct timespec tim;
  tim.tv_sec = 0;
  tim.tv_nsec = (long)sleep_time;
  nanosleep(&tim, NULL);
}

#if HAS_HBW

void *b_malloc(size_t size, bool use_hbw) {
  void *allocated;
  int err;
  if (use_hbw)
    err = hbw_posix_memalign(&allocated, LINE_SIZE, size);
  else
    err = posix_memalign(&allocated, LINE_SIZE, size);
  if (err) {
    std::cout << "Memory allocation error.\n";
    return 0;
  } else {
    return allocated;
  }
}

void b_free(void* ptr, bool use_hbw) {
  if (use_hbw)
    hbw_free(ptr);
  else
    free(ptr);
}

#else

void *b_malloc(size_t size, bool use_hbw) {
  void *allocated;
  //ignore use_hbw
  int err = posix_memalign(&allocated, LINE_SIZE, size);
  if (err) {
    std::cout << "Memory allocation error.\n";
    return 0;
  } else {
    return allocated;
  }
}

void b_free(void* ptr, bool use_hbw) {
  //ignore use_hbw
  free(ptr);
}

#endif

void raise_error(std::string error) {
  std::string message = "[E] " + error + "\n";
  std::cerr << message;
  exit(-1);
}

/* ================= MATRIX MEMORY MANAGEMENT ================= */

Matrix allocate_matrix(uint32_t rows, uint32_t columns, bool use_hbw) {
  uint64_t size = (uint64_t)rows * columns;
  uint32_t row;
  uint32_t reals_per_line = LINE_SIZE / sizeof(real);
  uint32_t padded_rows;
  if (data_rep == QUANTIZED)
    padded_rows = (rows + LINE_SIZE - 1) / LINE_SIZE * LINE_SIZE;
  else
    padded_rows = (rows + reals_per_line - 1)
        / reals_per_line * reals_per_line;
  size = (uint64_t)padded_rows * columns;
  Matrix mat = {};
  mat.columns = columns;
  mat.rows = rows;
  mat.padded_rows = padded_rows;
  mat.data = (real*)b_malloc(size * sizeof(real), use_hbw);
  if (!mat.data)
    raise_error("Matrix allocation failed.");
  return mat;
}

Matrix create_matrix(real *data, uint32_t rows, uint32_t columns,
    bool use_hbw) {
  Matrix mat = allocate_matrix(rows, columns, use_hbw);
  if (!data) {
    std::memset(mat.data, 0, sizeof(real) * (uint64_t)rows * columns);
  } else {
    for (uint32_t i = 0; i < columns; ++i) {
      std::memcpy(mat.data + i * (uint64_t)mat.padded_rows,
          data + i * (uint64_t)rows, rows * sizeof(real));
      for (uint32_t j = rows; j < mat.padded_rows; ++j)
        mat.data[i * (uint64_t)mat.padded_rows + j] = 0;
    }
  }
  return mat;
}

Matrix create_matrix(real *data, uint32_t rows, uint32_t columns) {
  return create_matrix(data, rows, columns, false);
}

Matrix create_matrix(uint32_t rows, uint32_t columns) {
  return create_matrix(nullptr, rows, columns, false);
}

Matrix create_matrix(uint32_t rows, uint32_t columns, bool use_hbw) {
  return create_matrix(nullptr, rows, columns, use_hbw);
}

Matrix read_matrix(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw) {
  Matrix mat = allocate_matrix(rows, columns, use_hbw);
  std::memset(mat.data, 0, (uint64_t)rows * columns * sizeof(real));
  real *x_buffer = new real[rows];
  for (uint32_t j = 0; j < columns; ++j) {
    x_stream.read(reinterpret_cast<char*>(x_buffer),
        std::streamsize(rows * sizeof(real)));
    std::memcpy(mat.data + j * (uint64_t)mat.padded_rows, x_buffer,
        rows * sizeof(real));
  }
  delete[] x_buffer;
  return mat;
}

void destroy(Matrix mat, bool use_hbw) {
  if (mat.data)
    b_free(mat.data, use_hbw);
  mat.data = nullptr;
}

void destroy(Matrix mat) {
  destroy(mat, false);
}

QuantMatrix allocate_matrix_quant(uint32_t rows, uint32_t columns,
    bool use_hbw) {
  uint32_t row;
  uint32_t reals_per_line = LINE_SIZE / sizeof(real);
  uint32_t padded_rows = (rows + LINE_SIZE - 1) / LINE_SIZE * LINE_SIZE;
  QuantMatrix mat = {};
  #if HAS_QUANTIZED
  mat.columns = columns;
  mat.rows = rows;
  mat.padded_rows = padded_rows;
  mat.data = (CloverVec*)b_malloc(columns * sizeof(CloverVec), use_hbw);
  for (uint32_t i = 0; i < columns; ++i)
    new(mat.data + i) CloverVec(padded_rows, use_hbw);
  if (!mat.data)
    raise_error("Matrix allocation failed.");
  for (uint32_t i = 0; i < columns; ++i)
    if (!mat.data[i].getData())
      raise_error("Matrix allocation failed.");
  #endif
  return mat;
}
  
QuantMatrix create_matrix_quant(real *data, uint32_t rows,
    uint32_t columns, bool use_hbw) {
  QuantMatrix mat = allocate_matrix_quant(rows, columns, use_hbw);
  #if HAS_QUANTIZED
  if (!data) {
    for (uint32_t i = 0; i < columns; ++i)
      mat.data[i].clear();
  } else {
    CloverVector32 buffer_vec(rows);
    float *buffer = buffer_vec.getData();
    for (uint32_t i=0; i<columns; ++i) {
      uint32_t row = i * rows;
      std::memcpy(buffer, data + i * rows,
          rows * sizeof(float));
      mat.data[i].quantize(buffer_vec);
    }
  }
  #endif
  return mat;
}

QuantMatrix create_matrix_quant(real *data, uint32_t rows,
    uint32_t columns) {
  return create_matrix_quant(data, rows, columns, false);
}

QuantMatrix create_matrix_quant(uint32_t rows, uint32_t columns) {
  return create_matrix_quant(nullptr, rows, columns, false);
}

QuantMatrix create_matrix_quant(uint32_t rows, uint32_t columns,
    bool use_hbw) {
  return create_matrix_quant(nullptr, rows, columns, use_hbw);
}

QuantMatrix read_matrix_quant(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw) {
  QuantMatrix mat = allocate_matrix_quant(rows, columns, use_hbw);
  #if HAS_QUANTIZED
  CloverVector32 buffer_vec(mat.padded_rows);
  float *buffer = buffer_vec.getData();
  float *buf2 = new float[rows];
  for (uint32_t i = 0; i < columns; ++i) {
    x_stream.read(reinterpret_cast<char*>(buf2),
        std::streamsize(rows * sizeof(float)));
    for (uint32_t j = 0; j < rows; ++j)
      buffer[j] = buf2[j];
    for (uint32_t j = rows; j < mat.padded_rows; ++j)
      buffer[j] = 0;
    mat.data[i].quantize(buffer_vec);
  }
  delete[] buf2;
  #endif
  return mat;
}

void destroy(QuantMatrix mat, bool use_hbw) {
  #if HAS_QUANTIZED
  if (mat.data) {
    for (uint32_t i = 0; i < mat.columns; ++i)
      mat.data[i].~CloverVec();
    b_free(mat.data, use_hbw);
  }
  mat.data = nullptr;
  #endif
}

void destroy(QuantMatrix mat) {
  destroy(mat, false);
}

SparseMatrix allocate_matrix_sparse(uint32_t rows, uint32_t columns,
    bool use_hbw) {
  SparseMatrix mat = {};
  mat.columns = columns;
  mat.rows = rows;
  mat.column_data = (SparseVector*)b_malloc(sizeof(SparseVector)
      * columns, use_hbw);
  if (!mat.column_data)
    raise_error("Matrix allocation failed.");
  return mat;   
}

SparseMatrix create_matrix_sparse(uint32_t rows, uint32_t columns,
    bool use_hbw) {
  SparseMatrix mat = allocate_matrix_sparse(rows, columns, use_hbw);
  for (uint32_t j = 0; j < columns; ++j) {
    mat.column_data[j].data = nullptr;
    mat.column_data[j].length = 0;
    mat.column_data[j].nnz = 0;
    mat.column_data[j].padded_nnz = 0;
    mat.column_data[j].max_idx = 0;
  }
  return mat;
}

SparseMatrix create_matrix_sparse(uint32_t rows, uint32_t columns) {
  return create_matrix_sparse(rows, columns, false);
}

SparseMatrix read_matrix_sparse(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw) {
  SparseMatrix mat = allocate_matrix_sparse(rows, columns, use_hbw);
  uint32_t nnz;
  uint32_t max_idx;
  uint32_t padded_nnz;
  uint32_t small_len;
  SparsePiece* piece_ptr;
  SparsePiece* prev_ptr;
  uint32_t* idx = new uint32_t[rows];
  float* vals = new float[rows];
  for (uint32_t j = 0; j < columns; ++j) {
    mat.column_data[j].data = nullptr;
    mat.column_data[j].max_idx = 0;
    piece_ptr = nullptr;
    prev_ptr = nullptr;
    nnz = 0;
    padded_nnz = 0;
    x_stream.read(reinterpret_cast<char*>(&nnz),
          std::streamsize(sizeof(uint32_t)));
    x_stream.read(reinterpret_cast<char*>(idx),
          std::streamsize(nnz * sizeof(uint32_t)));
    x_stream.read(reinterpret_cast<char*>(vals),
          std::streamsize(nnz * sizeof(float)));
    for (uint32_t i = 0; i < nnz; i += SPARSE_PIECE_LENGTH) {
      piece_ptr = (SparsePiece*)b_malloc(sizeof(SparsePiece), use_hbw);
      if (!piece_ptr)
        raise_error("Matrix allocation failed.");
      piece_ptr->next = nullptr;
      piece_ptr->values = (real*)b_malloc(
          SPARSE_PIECE_LENGTH * sizeof(real), use_hbw);
      piece_ptr->indices = (uint32_t*)b_malloc(
          SPARSE_PIECE_LENGTH * sizeof(uint32_t), use_hbw);
      if (!mat.column_data[j].data) {
        mat.column_data[j].data = piece_ptr;
      } else {
        prev_ptr->next = piece_ptr;
      }
      small_len = std::min((uint32_t)SPARSE_PIECE_LENGTH, nnz - i);
      std::memcpy(piece_ptr->values, vals + i,
          small_len * sizeof(float));
      std::memcpy(piece_ptr->indices, idx + i,
          small_len * sizeof(uint32_t));
      mat.column_data[j].max_idx = piece_ptr->indices[small_len - 1];
      piece_ptr->max_idx = piece_ptr->indices[small_len - 1];
      piece_ptr->small_len = small_len;
      prev_ptr = piece_ptr;
      padded_nnz += SPARSE_PIECE_LENGTH;
      if (small_len < SPARSE_PIECE_LENGTH) {
        std::memset(piece_ptr->values + small_len, 0,
            (SPARSE_PIECE_LENGTH - small_len) * sizeof(float));
        std::memset(piece_ptr->indices + small_len, 0,
            (SPARSE_PIECE_LENGTH - small_len) * sizeof(uint32_t));
      }
      if (i + SPARSE_PIECE_LENGTH >= nnz)
        max_idx = (small_len > 0
            ? piece_ptr->indices[small_len - 1] : 0);
    }
    mat.column_data[j].nnz = nnz;
    mat.column_data[j].length = rows;
    mat.column_data[j].padded_nnz = padded_nnz;
    mat.column_data[j].max_idx = max_idx;
  }
  delete[] idx;
  delete[] vals;
  return mat;
}

void destroy(SparseMatrix mat, bool use_hbw) {
  if (mat.column_data) {
    for (uint32_t j = 0; j < mat.columns; ++j) {
      SparsePiece *runner = mat.column_data[j].data;
      while (runner) {
        SparsePiece* ptr = runner;
        runner = runner->next;
        b_free(ptr->values, use_hbw);
        b_free(ptr->indices, use_hbw);
        b_free(ptr, use_hbw);
      }
    }
    b_free(mat.column_data, use_hbw);
  }
  mat.column_data = nullptr;
}

void destroy(SparseMatrix mat) {
  destroy(mat, false);
}

OneSparseMatrix read_matrix_one_sparse(std::ifstream &x_stream,
    uint32_t rows, uint32_t columns, bool use_hbw) {
  OneSparseMatrix mat = {};
  mat.columns = columns;
  mat.rows = rows;
  mat.column_data = (OneSparseVector*)b_malloc(sizeof(OneSparseVector)
      * columns, use_hbw);
  if (!mat.column_data)
    raise_error("Matrix allocation failed.");
  uint32_t nnz;
  uint32_t padded_nnz;
  uint32_t small_len;
  uint32_t* idx;
  float* vals;
  for (uint32_t j = 0; j < columns; ++j) {
    nnz = 0;
    padded_nnz = 0;
    x_stream.read(reinterpret_cast<char*>(&nnz),
          std::streamsize(sizeof(uint32_t)));
    padded_nnz = (nnz + 31) / 32 * 32;
    idx = (uint32_t*)b_malloc(padded_nnz * sizeof(uint32_t), use_hbw);
    vals = (float*)b_malloc(padded_nnz * sizeof(float), use_hbw);
    x_stream.read(reinterpret_cast<char*>(idx),
          std::streamsize(nnz * sizeof(uint32_t)));
    x_stream.read(reinterpret_cast<char*>(vals),
          std::streamsize(nnz * sizeof(float)));
    if (nnz != padded_nnz) {
      std::memset(vals + nnz, 0, (padded_nnz - nnz) * sizeof(float));
      std::memset(idx + nnz, 0, (padded_nnz - nnz) * sizeof(uint32_t));
    }
    mat.column_data[j].nnz = nnz;
    mat.column_data[j].length = rows;
    mat.column_data[j].padded_nnz = padded_nnz;
    mat.column_data[j].max_idx = (nnz > 0 ? idx[nnz - 1] : 0);
    mat.column_data[j].indices = idx;
    mat.column_data[j].values = vals;
  }
  return mat;
}

void destroy(OneSparseMatrix mat, bool use_hbw) {
  if (mat.column_data) {
    for (uint32_t j = 0; j < mat.columns; ++j) {
      b_free(mat.column_data[j].values, use_hbw);
      b_free(mat.column_data[j].indices, use_hbw);
    }
    b_free(mat.column_data, use_hbw);
  }
  mat.column_data = nullptr;
}

void destroy(OneSparseMatrix mat) {
  destroy(mat, false);
}

/* ================= VECTOR MEMORY MANAGEMENT ================= */

Vector allocate_vector(uint32_t length, bool use_hbw) {
  Vector vec;
  if (data_rep == QUANTIZED)
    vec.length = (length + LINE_SIZE - 1) / LINE_SIZE * LINE_SIZE;
  else
    vec.length = length;
  vec.data = (real*)b_malloc(vec.length * sizeof(real), use_hbw);
  if (!vec.data)
    raise_error("Vector allocation failed.");
  return vec;
}

Vector create_vector(real *data, uint32_t length, bool use_hbw) {
  Vector vec = allocate_vector(length, use_hbw);
  if (!data) {
    std::memset(vec.data, 0, sizeof(real) * vec.length);
  }
  else {
    std::memcpy(vec.data, data, sizeof(real) * length);
    if (vec.length != length)
      std::memset(vec.data + length, 0,
          sizeof(real) * (vec.length - length));
  }
  return vec;
}

Vector create_vector(real *data, uint32_t length) {
  return create_vector(data, length, false);
}

Vector create_vector(uint32_t length) {
  return create_vector(nullptr, length, false);
}

Vector create_vector(uint32_t length, bool use_hbw) {
  return create_vector(nullptr, length, use_hbw);
}

Vector read_vector(std::ifstream &y_stream,
    uint32_t length, bool use_hbw) {
  Vector vec = allocate_vector(length, use_hbw);
  y_stream.read(reinterpret_cast<char*>(vec.data),
      std::streamsize(length * sizeof(real)));
  if (vec.length != length)
      std::memset(vec.data + length, 0,
          sizeof(real) * (vec.length - length));
  return vec;
}

void destroy(Vector vec, bool use_hbw) {
  if (!vec.data)
    b_free(vec.data, use_hbw);
  vec.data = nullptr;
}

void destroy(Vector vec) {
  destroy(vec, false);
}

QuantVector allocate_vector_quant(uint32_t length, bool use_hbw) {
  QuantVector vec = {};
  #if HAS_QUANTIZED
  vec.length = (length + LINE_SIZE - 1) / LINE_SIZE * LINE_SIZE;
  uint32_t reals_per_line = LINE_SIZE / sizeof(real);
  vec.data = (CloverVec*)b_malloc(sizeof(CloverVec), use_hbw);
  new(vec.data) CloverVec(length, use_hbw);
  if (!vec.data || !vec.data->getData())
    raise_error("Vector allocation failed.");
  #endif
  return vec;
}

QuantVector create_vector_quant(real *data, uint32_t length,
    bool use_hbw) {
  QuantVector vec = allocate_vector_quant(length, use_hbw);
  #if HAS_QUANTIZED
  CloverVector32 buffer_vec(length);
  float *buffer = buffer_vec.getData();
  if (!data) {
    vec.data->clear();
  } else {
    std::memcpy(buffer, data, sizeof(float) * length);
    vec.data->quantize(buffer_vec);
  }
  #endif
  return vec;
}

QuantVector create_vector_quant(real *data, uint32_t length) {
  return create_vector_quant(data, length, false);
}

QuantVector create_vector_quant(uint32_t length) {
  return create_vector_quant(nullptr, length, false);
}

QuantVector create_vector_quant(uint32_t length, bool use_hbw) {
  return create_vector_quant(nullptr, length, use_hbw);
}

QuantVector read_vector_quant(std::ifstream &y_stream,
    uint32_t length, bool use_hbw) {
  QuantVector vec = allocate_vector_quant(length, use_hbw);
  #if HAS_QUANTIZED
  CloverVector32 buffer_vec(length);
  float *buffer = buffer_vec.getData();
  float *buf2 = new float[length];
  y_stream.read(reinterpret_cast<char*>(buf2),
      std::streamsize(length * sizeof(float)));
  for (uint32_t j = 0; j < length; ++j)
    buffer[j] = buf2[j];
  vec.data->quantize(buffer_vec);
  delete[] buf2;
  #endif
  return vec;
}

void destroy(QuantVector vec, bool use_hbw) {
  #if HAS_QUANTIZED
  if (!vec.data) {
    vec.data->~CloverVec();
    b_free(vec.data, use_hbw);
  }
  vec.data = nullptr;
  #endif
}

void destroy(QuantVector vec) {
  destroy(vec, false);
}

/* ================= GETTERS AND SETTERS ================= */

//Returns a vector containing pointer to a column,
//Modifications to the vector change the original matrix.
Vector get_column(Matrix mat, uint32_t index) {
  if (index >= mat.columns)
    raise_error("get_column: Matrix column index out of bounds! ("
        + std::to_string(index) + ">="
        + std::to_string(mat.columns) + ")");
  Vector vec = {};
  if (data_rep == QUANTIZED)
    vec.length = mat.padded_rows;
  else
    vec.length = mat.rows;
  vec.data = mat.data + ((uint64_t)mat.padded_rows * index);
  return vec;
}

QuantVector get_column(QuantMatrix mat, uint32_t index) {
  #if HAS_QUANTIZED
  if (index >= mat.columns)
    raise_error("get_column: Matrix column index out of bounds! ("
        + std::to_string(index) + ">="
        + std::to_string(mat.columns) + ")");
  QuantVector vec = {};
  vec.length = mat.padded_rows;
  vec.data = mat.data + index;
  return vec;
  #else
  QuantVector vec = {};
  return vec;
  #endif
}

SparseVector get_column(SparseMatrix mat, uint32_t index) {
  if (index >= mat.columns)
    raise_error("get_column: Matrix column index out of bounds! ("
        + std::to_string(index) + ">="
        + std::to_string(mat.columns) + ")");
  SparseVector vec = mat.column_data[index];
  return vec;
}

OneSparseVector get_column(OneSparseMatrix mat, uint32_t index) {
  if (index >= mat.columns)
    raise_error("get_column: Matrix column index out of bounds! ("
        + std::to_string(index) + ">="
        + std::to_string(mat.columns) + ")");
  OneSparseVector vec = mat.column_data[index];
  return vec;
}

real get_value(Vector vec, uint32_t index) {
  if (index >= vec.length)
    raise_error("get_value: Vector index out of bounds! ("
        + std::to_string(index) + ">="
        + std::to_string(vec.length) + ")");
  return vec.data[index];
}

real get_value(QuantVector vec, uint32_t index) {
  #if HAS_QUANTIZED
  if (index >= vec.length)
    raise_error("get_value: Vector index out of bounds! ("
        + std::to_string(index) + ">="
        + std::to_string(vec.length) + ")");
  return vec.data->get(index);
  #else
  return 0;
  #endif
}

void set_value(Vector &vec, uint32_t index, real value) {
  if (index >= vec.length)
    raise_error("get_value: Vector index out of bounds! ("
        + std::to_string(index) + ">="
        + std::to_string(vec.length) + ")");
  vec.data[index] = value;
}

void set(Vector &vec, Vector tvec) {
  if (vec.length != tvec.length)
    raise_error("set: Vector lengths do not match! ("
        + std::to_string(vec.length) + "!="
        + std::to_string(tvec.length) + ")");
  std::memcpy(vec.data, tvec.data, vec.length * sizeof(real));
}

void set_column(Matrix &mat, uint32_t column, Vector vec) {
  if (vec.length != mat.rows)
    raise_error("set_column: Vector does not match matrix column! ("
      + std::to_string(vec.length) + "!="
      + std::to_string(mat.rows) + ")");
  if (column >= mat.columns)
    raise_error("set_column: Matrix column index out of bounds! ("
      + std::to_string(column) + ">="
      + std::to_string(mat.columns) + ")");
  std::memcpy(mat.data + (uint64_t)mat.padded_rows * column, vec.data,
      vec.length * sizeof(real));
}

void set_column(QuantMatrix &mat, uint32_t column, QuantVector vec) {
  #if HAS_QUANTIZED
  if (vec.length != mat.padded_rows)
    raise_error("set_column: Vector does not match matrix column! ("
      + std::to_string(vec.length) + "!="
      + std::to_string(mat.padded_rows) + ")");
  if (column >= mat.columns)
    raise_error("set_column: Matrix column index out of bounds! ("
      + std::to_string(column) + ">="
      + std::to_string(mat.columns) + ")");
  std::memcpy(mat.data[column].getData(), vec.data->getData(),
      vec.data->getValueBytes());
  std::memcpy(mat.data[column].getScales(), vec.data->getScales(),
      vec.data->getScaleBytes());
  #endif
}

//PRE: mat column already has the required pieces
void set_column(SparseMatrix &mat, uint32_t column, SparseVector vec) {
  if (vec.length != mat.rows)
    raise_error("set_column: Vector does not match matrix column! ("
      + std::to_string(vec.length) + "!="
      + std::to_string(mat.rows) + ")");
  if (column >= mat.columns)
    raise_error("set_column: Matrix column index out of bounds! ("
      + std::to_string(column) + ">="
      + std::to_string(mat.columns) + ")");
  mat.column_data[column].nnz = vec.nnz;
  mat.column_data[column].padded_nnz = vec.padded_nnz;
  mat.column_data[column].length = vec.length;
  mat.column_data[column].max_idx = vec.max_idx;
  SparsePiece* source_ptr = vec.data;
  SparsePiece* target_ptr = mat.column_data[column].data;
  while (source_ptr && target_ptr) {
    std::memcpy(target_ptr->values, source_ptr->values,
        SPARSE_PIECE_LENGTH * sizeof(real));
    std::memcpy(target_ptr->indices, source_ptr->indices,
        SPARSE_PIECE_LENGTH * sizeof(uint32_t));
    target_ptr->small_len = source_ptr->small_len;
    source_ptr = source_ptr->next;
    target_ptr = target_ptr->next;
  }
  if (source_ptr || target_ptr)
    raise_error("set_column: Vector pieces do not match the column!");
}

//PRE: mat column already has the required pieces
void set_column(SparseMatrix &mat, uint32_t column,
    OneSparseVector vec) {
  if (vec.length != mat.rows)
    raise_error("set_column: Vector does not match matrix column! ("
      + std::to_string(vec.length) + "!="
      + std::to_string(mat.rows) + ")");
  if (column >= mat.columns)
    raise_error("set_column: Matrix column index out of bounds! ("
      + std::to_string(column) + ">="
      + std::to_string(mat.columns) + ")");
  mat.column_data[column].nnz = vec.nnz;
  mat.column_data[column].length = vec.length;
  mat.column_data[column].max_idx = vec.max_idx;
  SparsePiece* target_ptr = mat.column_data[column].data;
  uint32_t source_idx = 0;
  uint32_t small_len;
  uint32_t padded_nnz = 0;
  while (target_ptr && source_idx < vec.nnz) {
    small_len = std::min((uint32_t)SPARSE_PIECE_LENGTH,
        vec.nnz - source_idx);
    std::memcpy(target_ptr->values, vec.values + source_idx,
        small_len * sizeof(float));
    std::memcpy(target_ptr->indices, vec.indices + source_idx,
        small_len * sizeof(uint32_t));
    if (small_len != SPARSE_PIECE_LENGTH) {
      std::memset(target_ptr->values + small_len, 0,
          (SPARSE_PIECE_LENGTH - small_len) * sizeof(float));
      std::memset(target_ptr->indices + small_len, 0,
          (SPARSE_PIECE_LENGTH - small_len) * sizeof(uint32_t));
    }
    target_ptr->small_len = small_len;
    target_ptr->max_idx = (small_len > 0
        ? target_ptr->values[small_len - 1] : 0);
    target_ptr = target_ptr->next;
    source_idx += SPARSE_PIECE_LENGTH;
  }
  mat.column_data[column].padded_nnz = source_idx;
  if (target_ptr || source_idx < vec.nnz)
    raise_error("set_column: Vector pieces do not match the column!");
}

/* ================= SCALAR OPERATIONS ================= */

void scalar_multiply(Vector target, Vector vec, real scalar) {
  if (target.length != vec.length)
    raise_error("scalar_multiply: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  for (uint32_t i = 0; i < vec.length; ++i)
    target.data[i] = vec.data[i] * scalar;
}

void scalar_divide(Vector target, Vector vec, real scalar) {
  if (target.length != vec.length)
    raise_error("scalar_divide: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  for (uint32_t i = 0; i < vec.length; ++i)
    target.data[i] = vec.data[i] / scalar;
}

real norm_2_squared(Vector vec) {
  return dot_product(vec, vec);
}

real norm_1(Vector vec) {
  real sum1 = 0;
  real sum2 = 0;
  real sum3 = 0;
  real sum4 = 0;
  real sum;
  uint32_t len_div_4 = vec.length - (vec.length & 3);
  for (uint32_t i = 0; i < len_div_4; i += 4) {
    sum1 += std::abs(vec.data[i]);
    sum2 += std::abs(vec.data[i+1]);
    sum3 += std::abs(vec.data[i+2]);
    sum4 += std::abs(vec.data[i+3]);
  }
  sum = sum1 + sum2 + sum3 + sum4;
  for (uint32_t i = len_div_4; i < vec.length; ++i)
    sum += std::abs(vec.data[i]);
  return sum;
}

real dot_product(Vector vec1, Vector vec2) {
  return dot_product(vec1, vec2, 0, vec1.length);
}

real dot_product(Vector vec1, Vector vec2,
    uint32_t start, uint32_t end) {
  if (vec1.length != vec2.length)
    raise_error("dot_product: Vector lengths are not equal! ("
      + std::to_string(vec1.length) + "!="
      + std::to_string(vec2.length) + ")");
  if (end > vec1.length)
    raise_error("dot_product: Range out of bounds! ("
      + std::to_string(end) + ">" + std::to_string(vec1.length) + ")");
  real sum1 = 0;
  real sum2 = 0;
  real sum3 = 0;
  real sum4 = 0;
  real sum;
  uint32_t end_div_4 = end - ((end - start) & 3);
  for (uint32_t i = start; i < end_div_4; i += 4) {
    sum1 += vec1.data[i] * vec2.data[i];
    sum2 += vec1.data[i+1] * vec2.data[i+1];
    sum3 += vec1.data[i+2] * vec2.data[i+2];
    sum4 += vec1.data[i+3] * vec2.data[i+3];
  }
  sum = sum1 + sum2 + sum3 + sum4;
  for (uint32_t i = end_div_4; i < end; ++i)
    sum += vec1.data[i] * vec2.data[i];
  return sum;
}

void scalar_multiply_add(Vector target, Vector vec, real scalar) {
  scalar_multiply_add(target, vec, scalar, 0, target.length);
}

void scalar_multiply_add(Vector target, Vector vec, real scalar,
    uint32_t start, uint32_t end) {
  if (target.length != vec.length)
    raise_error("scalar_multiply_add: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  if (end > vec.length)
    raise_error("scalar_multiply_add: Range out of bounds! ("
        + std::to_string(end) + ">" + std::to_string(vec.length) + ")");
  for (uint32_t i = start; i < end; ++i)
    target.data[i] += vec.data[i] * scalar;
}

real dot_product(SparseVector vec1, Vector vec2) {
  return dot_product(vec2, vec1, 0, vec1.length);
}

real dot_product(Vector vec1, SparseVector vec2) {
  return dot_product(vec1, vec2, 0, vec1.length);
}

real dot_product(SparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end) {
  return dot_product(vec2, vec1, start, end);
}

real dot_product(Vector vec1, SparseVector vec2,
    uint32_t start, uint32_t end) {
  if (vec1.length != vec2.length)
    raise_error("dot_product: Vector lengths are not equal! ("
      + std::to_string(vec1.length) + "!="
      + std::to_string(vec2.length) + ")");
  if (end > vec1.length)
    raise_error("dot_product: Range out of bounds! ("
      + std::to_string(end) + ">" + std::to_string(vec1.length) + ")");
  SparsePiece* ptr = vec2.data;
  real sum = 0;
  uint32_t large_ctr = vec2.nnz;
  uint32_t ith_piece = 0;
    
  if (vec2.max_idx >= start && ptr && ptr->indices[0] < end) {

    while (ptr && ptr->next && ptr->next->indices[0] <= start) {
      ptr = ptr->next;
      ++ith_piece;
    }
    while (ptr && ptr->next) {
      uint32_t ctr = 0;
      uint32_t small_ctr = SPARSE_PIECE_LENGTH;
      while (ctr < small_ctr && ptr->indices[ctr] < start)
        ++ctr;
      while (ctr < small_ctr && ptr->indices[ctr] < end) {
        sum += vec1.data[ptr->indices[ctr]] * ptr->values[ctr];
        ++ctr;
      }
      if (ctr < small_ctr) return sum;
      ptr = ptr->next;
      ++ith_piece;
    }
    if (ptr) {
      uint32_t ctr = 0;
      uint32_t small_ctr = large_ctr - ith_piece * SPARSE_PIECE_LENGTH;
      while (ctr < small_ctr && ptr->indices[ctr] < start)
          ++ctr;
      while (ctr < small_ctr && ptr->indices[ctr] < end) {
        sum += vec1.data[ptr->indices[ctr]] * ptr->values[ctr];
        ++ctr;
      }
    }
  }
  return sum;
}

void scalar_multiply_add(Vector target, SparseVector vec, real scalar) {
  scalar_multiply_add(target, vec, scalar, 0, target.length);
}

void scalar_multiply_add(Vector target, SparseVector vec, real scalar,
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

  if (vec.max_idx >= start && ptr && ptr->indices[0] < end) {

    while (ptr && ptr->next && ptr->next->indices[0] <= start) {
      ptr = ptr->next;
      ++ith_piece;
    }
    while (ptr && ptr->next) {
      uint32_t ctr = 0;
      uint32_t small_ctr = SPARSE_PIECE_LENGTH;
      while (ctr < small_ctr && ptr->indices[ctr] < start)
        ++ctr;
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
      uint32_t small_ctr = large_ctr - ith_piece * SPARSE_PIECE_LENGTH;
      while (ctr < small_ctr && ptr->indices[ctr] < start)
          ++ctr;
      while (ctr < small_ctr && ptr->indices[ctr] < end) {
        target.data[ptr->indices[ctr]] += ptr->values[ctr] * scalar;
        ++ctr;
      }
    }
  }
}

real norm_2_squared(SparseVector vec) {
  SparsePiece* ptr = vec.data;
  uint32_t large_ctr = vec.nnz;
  uint32_t ith_piece = 0;
  real sum = 0;
  while (ptr) {
    uint32_t ctr = 0;
    uint32_t small_ctr = ((ptr->next)
        ? SPARSE_PIECE_LENGTH
        : large_ctr - ith_piece * SPARSE_PIECE_LENGTH);
    for (uint32_t ctr = 0; ctr < small_ctr; ++ctr)
      sum += ptr->values[ctr] * ptr->values[ctr];
    ptr = ptr->next;
    ++ith_piece;
  }
  return sum;
}

real dot_product(OneSparseVector vec1, Vector vec2) {
  return dot_product(vec2, vec1, 0, vec1.length);
}

real dot_product(Vector vec1, OneSparseVector vec2) {
  return dot_product(vec1, vec2, 0, vec1.length);
}

real dot_product(OneSparseVector vec1, Vector vec2,
    uint32_t start, uint32_t end) {
  return dot_product(vec2, vec1, start, end);
}

real dot_product(Vector vec1, OneSparseVector vec2,
    uint32_t start, uint32_t end) {
  if (vec1.length != vec2.length)
    raise_error("dot_product: Vector lengths are not equal! ("
      + std::to_string(vec1.length) + "!="
      + std::to_string(vec2.length) + ")");
  if (end > vec1.length)
    raise_error("dot_product: Range out of bounds! ("
      + std::to_string(end) + ">" + std::to_string(vec1.length) + ")");
  real sum = 0;
  uint32_t i = 0;
  while (i < vec2.nnz && vec2.indices[i] < start)
    ++i;
  while (i < vec2.nnz && vec2.indices[i] < end) {
    sum += vec2.values[i] * vec1.data[vec2.indices[i]];
    ++i;
  }
  return sum;
}

void scalar_multiply_add(Vector target, OneSparseVector vec,
    real scalar) {
  scalar_multiply_add(target, vec, scalar, 0, target.length);
}

void scalar_multiply_add(Vector target, OneSparseVector vec,
    real scalar, uint32_t start, uint32_t end) {
  if (target.length != vec.length)
    raise_error("scalar_multiply_add: Vector lengths do not match! ("
        + std::to_string(target.length) + "!="
        + std::to_string(vec.length) + ")");
  if (end > vec.length)
    raise_error("scalar_multiply_add: Range out of bounds! ("
        + std::to_string(end) + ">" + std::to_string(vec.length) + ")");
  uint32_t i = 0;
  while (i < vec.nnz && vec.indices[i] < start)
    ++i;
  while (i < vec.nnz && vec.indices[i] < end) {
    target.data[vec.indices[i]] += vec.values[i] * scalar;
    ++i;
  }
}

real norm_2_squared(OneSparseVector vec) {
  real sum = 0;
  for (uint32_t i = 0; i < vec.nnz; ++i)
    sum += vec.values[i] * vec.values[i];
  return sum;
}

real dot_product(QuantVector vec1, QuantVector vec2,
    uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  CloverVector32 vec32_1(vec1.length);
  CloverVector32 vec32_2(vec2.length);
  vec1.data->restore(vec32_1);
  vec2.data->restore(vec32_2);
  Vector v1 = {vec32_1.getData(), vec1.length};
  Vector v2 = {vec32_2.getData(), vec2.length};
  return dot_product(v1, v2, start, end);
  #else
  return 0;
  #endif
}

real dot_product(Vector vec1, QuantVector vec2,
    uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  CloverVector32 vec32_2(vec2.length);
  vec2.data->restore(vec32_2);
  Vector v2 = {vec32_2.getData(), vec2.length};
  return dot_product(vec1, v2, start, end);
  #else
  return 0;
  #endif
}

void scalar_multiply_add(QuantVector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  CloverVector32 vec32_1(target.length);
  CloverVector32 vec32_2(vec.length);
  target.data->restore(vec32_1);
  vec.data->restore(vec32_2);
  Vector v1 = {vec32_1.getData(), target.length};
  Vector v2 = {vec32_2.getData(), vec.length};
  scalar_multiply_add(v1, v2, scalar, start, end);
  target.data->quantize(vec32_1);
  #endif
}

void scalar_multiply_add(Vector target, QuantVector vec,
    real scalar, uint32_t start, uint32_t end) {
  #if HAS_QUANTIZED
  CloverVector32 vec32_2(vec.length);
  vec.data->restore(vec32_2);
  Vector v2 = {vec32_2.getData(), vec.length};
  scalar_multiply_add(target, v2, scalar, start, end);
  #endif
}

real dot_product(QuantVector vec1, QuantVector vec2) {
  #if HAS_QUANTIZED
  return dot_product(vec1, vec2, 0, vec1.length);
  #else
  return 0;
  #endif
}

real dot_product(Vector vec1, QuantVector vec2) {
  #if HAS_QUANTIZED
  return dot_product(vec1, vec2, 0, vec1.length);
  #else
  return 0;
  #endif
}

void scalar_multiply_add(QuantVector target, QuantVector vec,
    real scalar) {
  #if HAS_QUANTIZED   
  scalar_multiply_add(target, vec, scalar, 0, vec.length);
  #endif
}

real norm_2_squared(QuantVector vec) {
  #if HAS_QUANTIZED
  CloverVector32 vec32_1(vec.length);
  vec.data->restore(vec32_1);
  Vector v1 = {vec32_1.getData(), vec.length};
  return norm_2_squared(v1);
  #else
  return 0;
  #endif
}

void scalar_multiply(QuantVector target, Vector vec, real scalar) {
  #if HAS_QUANTIZED
  CloverVector32 vec32(target.length);
  Vector temp = {vec32.getData(), target.length};
  scalar_multiply(temp, vec, scalar);
  target.data->quantize(vec32);
  #endif
}

void scalar_divide(QuantVector target, Vector vec, real scalar) {
  #if HAS_QUANTIZED
  CloverVector32 vec32(target.length);
  Vector temp = {vec32.getData(), target.length};
  scalar_divide(temp, vec, scalar);
  target.data->quantize(vec32);
  #endif
}

/* ================= VARIOUS HELPERS ================= */

void set_zero(Vector &vec) {
  std::memset(vec.data, 0, vec.length * sizeof(real));
}

void set_zero(QuantVector &vec) {
  #if HAS_QUANTIZED
  int_quant* vec_data = vec.data->getData();
  vec.data->clear();
  #endif
}

/* ================= QUANTIZED HELPERS ================= */

#if HAS_QUANTIZED
#if !SCALAR
inline void _mm256_transpose8_ps(
    __m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3,
    __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7
){
  __m256 u0, u1, u2, u3, u4, u5, u6, u7;
  __m256 s0, s1, s2, s3, s4, s5, s6, s7;
  
  u0 = _mm256_unpacklo_ps(r0, r1);
  u1 = _mm256_unpackhi_ps(r0, r1);
  u2 = _mm256_unpacklo_ps(r2, r3);
  u3 = _mm256_unpackhi_ps(r2, r3);
  u4 = _mm256_unpacklo_ps(r4, r5);
  u5 = _mm256_unpackhi_ps(r4, r5);
  u6 = _mm256_unpacklo_ps(r6, r7);
  u7 = _mm256_unpackhi_ps(r6, r7);

  s0 = _mm256_shuffle_ps(u0,u2,_MM_SHUFFLE(1,0,1,0));
  s1 = _mm256_shuffle_ps(u0,u2,_MM_SHUFFLE(3,2,3,2));
  s2 = _mm256_shuffle_ps(u1,u3,_MM_SHUFFLE(1,0,1,0));
  s3 = _mm256_shuffle_ps(u1,u3,_MM_SHUFFLE(3,2,3,2));
  s4 = _mm256_shuffle_ps(u4,u6,_MM_SHUFFLE(1,0,1,0));
  s5 = _mm256_shuffle_ps(u4,u6,_MM_SHUFFLE(3,2,3,2));
  s6 = _mm256_shuffle_ps(u5,u7,_MM_SHUFFLE(1,0,1,0));
  s7 = _mm256_shuffle_ps(u5,u7,_MM_SHUFFLE(3,2,3,2));

  r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
  r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
  r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
  r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
  r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
  r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
  r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
  r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

void inner_transpose(float *v, uint32_t end) {
  uint32_t i;
  __m256 u1, u2, u3, u4, u5, u6, u7, u8;
  for (i = 0; i < end; i += 64) {
    float* vi = v + i;
    u1 = _mm256_loadu_ps(vi);
    u2 = _mm256_loadu_ps(vi + 8);
    u3 = _mm256_loadu_ps(vi + 16);
    u4 = _mm256_loadu_ps(vi + 24);
    u5 = _mm256_loadu_ps(vi + 32);
    u6 = _mm256_loadu_ps(vi + 40);
    u7 = _mm256_loadu_ps(vi + 48);
    u8 = _mm256_loadu_ps(vi + 56);
    _mm256_transpose8_ps(u1, u2, u3, u4, u5, u6, u7, u8);
    _mm256_storeu_ps(vi, u1);
    _mm256_storeu_ps(vi + 8, u2);
    _mm256_storeu_ps(vi + 16, u3);
    _mm256_storeu_ps(vi + 24, u4);
    _mm256_storeu_ps(vi + 32, u5);
    _mm256_storeu_ps(vi + 40, u6);
    _mm256_storeu_ps(vi + 48, u7);
    _mm256_storeu_ps(vi + 56, u8);
  }
}
#endif
#endif

void inner_transpose(Vector vec) {
  #if HAS_QUANTIZED
  #if !SCALAR
  uint32_t end = vec.length - (vec.length & 63);
  uint32_t i;
  float *v = vec.data;
  __m256 u1, u2, u3, u4, u5, u6, u7, u8;
  for (i = 0; i < end; i += 64) {
    float* vi = v + i;
    u1 = _mm256_loadu_ps(vi);
    u2 = _mm256_loadu_ps(vi + 8);
    u3 = _mm256_loadu_ps(vi + 16);
    u4 = _mm256_loadu_ps(vi + 24);
    u5 = _mm256_loadu_ps(vi + 32);
    u6 = _mm256_loadu_ps(vi + 40);
    u7 = _mm256_loadu_ps(vi + 48);
    u8 = _mm256_loadu_ps(vi + 56);
    _mm256_transpose8_ps(u1, u2, u3, u4, u5, u6, u7, u8);
    _mm256_storeu_ps(vi, u1);
    _mm256_storeu_ps(vi + 8, u2);
    _mm256_storeu_ps(vi + 16, u3);
    _mm256_storeu_ps(vi + 24, u4);
    _mm256_storeu_ps(vi + 32, u5);
    _mm256_storeu_ps(vi + 40, u6);
    _mm256_storeu_ps(vi + 48, u7);
    _mm256_storeu_ps(vi + 56, u8);
  }
  #endif
  #endif
}
