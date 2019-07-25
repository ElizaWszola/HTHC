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

#ifndef PIECE_POOL_H
#define PIECE_POOL_H

#include "algebra.h"

class PiecePool {
    private:
    
    SparsePiece* stack;
    uint64_t capacity;
    
    public:
    
    //allocate as many pieces as possible and add them to the stack
    void allocate(uint64_t available_size, bool hbw);
    void deallocate(bool hbw);
    
    SparsePiece* pop();
    void push(SparsePiece* ptr);

    uint64_t get_capacity(); //how many pieces can I afford?
    uint64_t get_pieces(); //how many pieces left?
    
};

#endif
