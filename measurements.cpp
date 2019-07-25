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

#include "measurements.h"

int64_t get_time_difference(clocktime *start, clocktime *end) {
  int64_t seconds_diff = (int64_t)(end->tv_sec)
      - (int64_t)(start->tv_sec);
  int64_t end_nano = end->tv_nsec;
  int64_t start_nano = start->tv_nsec;
  if (start_nano <= end_nano)
    return seconds_diff * GIGA + end_nano - start_nano;
  return (seconds_diff - 1) * GIGA + GIGA - (start_nano - end_nano);
}

int get_time(clocktime *time) {
    return clock_gettime(CLOCK_REALTIME, time);
}
