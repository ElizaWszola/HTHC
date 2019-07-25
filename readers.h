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

#ifndef READERS_H
#define READERS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include "algebra.h"

Matrix read_matrix_from_binary_dual(std::string file_name,
    uint32_t samples, uint32_t features, bool use_hbw);
Matrix read_matrix_from_binary_primal(std::string file_name,
    uint32_t samples, uint32_t features, bool use_hbw);
QuantMatrix read_quantized_matrix_from_binary_dual(
    std::string file_name, uint32_t samples, uint32_t features,
    bool use_hbw);
QuantMatrix read_quantized_matrix_from_binary_primal(
    std::string file_name, uint32_t samples, uint32_t features,
    bool use_hbw);
OneSparseMatrix read_sparse_matrix_from_binary_dual(
    std::string file_name, uint32_t samples, uint32_t features,
    bool use_hbw);
OneSparseMatrix read_sparse_matrix_from_binary_primal(
    std::string file_name, uint32_t samples, uint32_t features,
    bool use_hbw);
Vector read_vector_from_binary(std::string file_name,
    uint32_t length, bool use_hbw);
QuantVector read_quantized_vector_from_binary(std::string file_name,
    uint32_t length, bool use_hbw);

#endif
