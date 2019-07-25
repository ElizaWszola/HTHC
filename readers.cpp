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

#include "readers.h"

Matrix read_matrix_from_binary(std::string file_name,
    uint32_t rows, uint32_t columns, bool use_hbw) {
  std::ifstream x_stream(file_name + "X",
      std::ios::in | std::ios::binary);
  if (x_stream.is_open()) {
    Matrix mat = read_matrix(x_stream, rows, columns, use_hbw);
    x_stream.close();
    return mat;
  } else {
    std::cout << "Failed to locate " << file_name << "X"
        << "." << std::endl;
    return {};
  }
}

Matrix read_matrix_from_binary_dual(std::string file_name,
    uint32_t samples, uint32_t features, bool use_hbw) {
  return read_matrix_from_binary(file_name + "_dual",
      features, samples, use_hbw);   
}

Matrix read_matrix_from_binary_primal(std::string file_name,
    uint32_t samples, uint32_t features, bool use_hbw) {
  return read_matrix_from_binary(file_name, samples, features, use_hbw);
}

Vector read_vector_from_binary(std::string file_name,
    uint32_t length, bool use_hbw) {
  std::ifstream y_stream(file_name + "Y",
      std::ios::in | std::ios::binary);
  if (y_stream.is_open()) {
    Vector vec = read_vector(y_stream, length, use_hbw);
    y_stream.close();
    return vec;
  } else {
    std::cout << "Failed to locate " << file_name << "Y"
        << "." << std::endl;
    return {};
  }
}

QuantMatrix read_quantized_matrix_from_binary(std::string file_name,
    uint32_t rows, uint32_t columns, bool use_hbw) {
  std::ifstream x_stream(file_name + "X",
      std::ios::in | std::ios::binary);
  if (x_stream.is_open()) {
    QuantMatrix mat = read_matrix_quant(x_stream, rows, columns,
        use_hbw);
    x_stream.close();
    return mat;
  } else {
    std::cout << "Failed to locate " << file_name << "X"
        << "." << std::endl;
    return {};
  }
}

QuantMatrix read_quantized_matrix_from_binary_dual(
    std::string file_name, uint32_t samples, uint32_t features,
    bool use_hbw) {
  return read_quantized_matrix_from_binary(file_name + "_dual",
      features, samples, use_hbw);   
}

QuantMatrix read_quantized_matrix_from_binary_primal(
    std::string file_name, uint32_t samples, uint32_t features,
    bool use_hbw) {
  return read_quantized_matrix_from_binary(file_name, samples, features,
      use_hbw);
}

QuantVector read_quantized_vector_from_binary(std::string file_name,
    uint32_t length, bool use_hbw) {
  std::ifstream y_stream(file_name + "Y",
      std::ios::in | std::ios::binary);
  if (y_stream.is_open()) {
    QuantVector vec = read_vector_quant(y_stream, length, use_hbw);
    y_stream.close();
    return vec;
  } else {
    std::cout << "Failed to locate " << file_name << "Y"
        << "." << std::endl;
    return {};
  }
}

OneSparseMatrix read_sparse_matrix_from_binary(std::string file_name,
    uint32_t rows, uint32_t columns, bool use_hbw) {
  std::ifstream x_stream(file_name + "X",
      std::ios::in | std::ios::binary);
  if (x_stream.is_open()) {
    OneSparseMatrix mat = read_matrix_one_sparse(x_stream,
        rows, columns, use_hbw);
    x_stream.close();
    return mat;
  } else {
    std::cout << "Failed to locate " << file_name << "X"
        << "." << std::endl;
    return {};
  }
}

OneSparseMatrix read_sparse_matrix_from_binary_dual(
    std::string file_name, uint32_t samples, uint32_t features,
    bool use_hbw) {
  return read_sparse_matrix_from_binary(file_name + "_dual_sparse",
      features, samples, use_hbw);   
}

OneSparseMatrix read_sparse_matrix_from_binary_primal(
    std::string file_name,
    uint32_t samples, uint32_t features, bool use_hbw) {
  return read_sparse_matrix_from_binary(file_name + "_sparse",
      samples, features, use_hbw);
}
