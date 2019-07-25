cmake_minimum_required (VERSION 2.8.1)
project (HTHC)
include_directories("./" "./include")
set(CMAKE_VERBOSE_MAKEFILE, ON)

#if no ICC available, set this to g++ and disable quantized below
set(CMAKE_CXX_COMPILER "icpc")

#important variables for troubleshooting and playing around
#by default, these variables should be set to false
set(DISABLE_AVX512 false)     #disable explicit AVX-512 vectorization
set(DISABLE_HBW false)        #disable high-bandwidth memory
set(DISABLE_LOCKS false)      #disable locks (use "HOGWILD!"-style updates)
set(DISABLE_QUANTIZED false)  #disable quantized type (helps if no Intel tools available)
#end if important variables

if(DISABLE_AVX512)
  add_definitions(-DSCALAR=true)
  if(${CMAKE_CXX_COMPILER} MATCHES "icpc") 
    set(AVX_FLAG "-xCORE-AVX2")
  else()
    set(AVX_FLAG "-mavx2")
  endif(${CMAKE_CXX_COMPILER} MATCHES "icpc")
else()
  add_definitions(-DSCALAR=false)
  if(${CMAKE_CXX_COMPILER} MATCHES "icpc") 
    set(AVX_FLAG "-xCOMMON-AVX512")
  else()
    set(AVX_FLAG "-mavx512f")
  endif(${CMAKE_CXX_COMPILER} MATCHES "icpc")
endif(DISABLE_AVX512)

if(DISABLE_HBW)
  add_definitions(-DHAS_HBW=false)
  set(MEMKIND_FLAG "")
else()
  add_definitions(-DHAS_HBW=true)
  set(MEMKIND_FLAG "-lmemkind")
endif(DISABLE_HBW)

if(DISABLE_LOCKS)
  add_definitions(-DLOCK=false)
else()
  add_definitions(-DLOCK=true)
endif(DISABLE_LOCKS)

if(DISABLE_QUANTIZED)
  add_definitions(-DHAS_QUANTIZED=false)
  set(MKL_FLAG "")
  set(IPP_FLAG "")
else()
  add_definitions(-DHAS_QUANTIZED=true)
  set(MKL_FLAG "-mkl")
  set(IPP_FLAG "-ipp")
endif(DISABLE_QUANTIZED)

if(${CMAKE_CXX_COMPILER} MATCHES "icpc")
  set(OMP_FLAG "-qopenmp")
else()
  add_definitions(-DREPLACE_INTRINSICS=true)
  set(OMP_FLAG "-fopenmp")
endif(${CMAKE_CXX_COMPILER} MATCHES "icpc") 

set(CMAKE_CXX_FLAGS "-std=c++11 -pthread -O2 ${OMP_FLAG} ${MKL_FLAG} ${IPP_FLAG} ${MEMKIND_FLAG} ${AVX_FLAG}")
set(SOURCE_FILES "main.cpp" "task_a.cpp" "task_b.cpp" "algebra.cpp" "threaded.cpp" "measurements.cpp" "vectorized.cpp" "readers.cpp" "piece_pool.cpp")
add_executable(hthc ${SOURCE_FILES})
