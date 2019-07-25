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

#include "piece_pool.h"

void PiecePool::allocate(uint64_t available_length, bool hbw) {
  stack = nullptr;
  uint64_t pieces = available_length;
  capacity = available_length;
  for (uint64_t i = 0; i < pieces; ++i) {
    SparsePiece *ptr = (SparsePiece*)b_malloc(sizeof(SparsePiece), hbw);
    ptr->next = stack;
    stack = ptr;
  }
  SparsePiece *runner = stack;
  while (runner) {
    #if HAS_HBW
    if(hbw) {
      hbw_posix_memalign((void**)&(runner->values), LINE_SIZE,
          SPARSE_PIECE_LENGTH * sizeof(real));
      hbw_posix_memalign((void**)&(runner->indices), LINE_SIZE,
          SPARSE_PIECE_LENGTH * sizeof(uint32_t));
    } else {
      posix_memalign((void**)&(runner->values), LINE_SIZE,
          SPARSE_PIECE_LENGTH * sizeof(real));
      posix_memalign((void**)&(runner->indices), LINE_SIZE,
          SPARSE_PIECE_LENGTH * sizeof(uint32_t));
    }
    #else
    posix_memalign((void**)&(runner->values), LINE_SIZE,
        SPARSE_PIECE_LENGTH * sizeof(real));
    posix_memalign((void**)&(runner->indices), LINE_SIZE,
        SPARSE_PIECE_LENGTH * sizeof(uint32_t));
    #endif
    runner = runner->next;
  }
}

void PiecePool::deallocate(bool hbw) {
  SparsePiece *runner = stack;
  while (runner) {
    SparsePiece* ptr = runner;
    runner = runner->next;
    #if HAS_HBW
    if (hbw) {
      hbw_free(ptr->values);
      hbw_free(ptr->indices);
    } else {
      free(ptr->values);
      free(ptr->indices);
    }
    #else
    free(ptr->values);
    free(ptr->indices);
    #endif
    b_free(ptr, hbw);
  }
  stack = nullptr;
}

SparsePiece* PiecePool::pop() {
  SparsePiece* ptr = stack;
  if (ptr == nullptr) {
    std::cout
        << "Null pointer returned from piece pool! (Rest In Pieces)\n";
    exit(-1);
  } else {
    stack = ptr->next;
    ptr->next = nullptr;
  }
  return ptr;
}

void PiecePool::push(SparsePiece* ptr) {
  ptr->next = stack;
  stack = ptr;
}

uint64_t PiecePool::get_capacity() {
  return capacity;
}

uint64_t PiecePool::get_pieces() {
  uint64_t pieces = 0;
  SparsePiece *runner = stack;
  while (runner) {
    ++pieces;
    runner = runner->next;
  }
  return pieces;
}
