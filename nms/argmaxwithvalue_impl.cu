/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"

#include "backend/kernel_compiler/gpu/cuda_impl/topk_lib.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/argmaxwithvalue_impl.cuh"
#include <limits>
#include <algorithm>

// Kernel started from here
#define L2_ARGMAX_HELPER(BLOCK, NUM_WARP_Q, NUM_THREAD_Q, IS_DESCEND)                                              \
  do {                                                                                                             \
    L2ArgMax<T, S, NUM_WARP_Q, NUM_THREAD_Q, BLOCK, IS_DESCEND>                                                    \
      <<<GET_BLOCKS((BLOCK * block_num_limit)), BLOCK, 0, stream>>>(outer_size, inner_size, stride, input, output, \
                                                                    output_index);                                 \
  } while (0)

#define ARG_LEFT_INSERT_THREAD_QUEUE(_k, _v)                                    \
  do {                                                                          \
    if (is_descend ? Cmp<T>::gt(_k, warp_K_top) : Cmp<T>::lt(_k, warp_K_top)) { \
      {                                                                         \
        _Pragma("unroll") for (int i = thread_queue - 1; i > 0; --i) {          \
          threadK[i] = threadK[i - 1];                                          \
          threadV[i] = threadV[i - 1];                                          \
        }                                                                       \
      }                                                                         \
      threadK[0] = _k;                                                          \
      threadV[0] = _v;                                                          \
      ++num_vals;                                                               \
    }                                                                           \
  } while (0)

template <typename T, typename S, int warp_queue, int thread_queue, int threads_per_block, bool is_descend>
__global__ void L2ArgMax(int outer_size, int inner_size, int stride, const T *input, T *output, S *output_index) {
  constexpr int kNumWarps = threads_per_block / kWarpSize;
  const T init_K = static_cast<T>(-9999);
  constexpr S init_V = static_cast<S>(-1);

  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < blockDim.x * outer_size * stride;
       t_idx += blockDim.x * gridDim.x) {
    __shared__ T shared_K[kNumWarps * warp_queue];
    __shared__ S shared_V[kNumWarps * warp_queue];

    T threadK[thread_queue];  // NOLINT
    S threadV[thread_queue];  // NOLINT

    T *warp_K;
    S *warp_V;

    T warp_K_top = init_K;
    int k_minus_1 = 0;
    int num_vals = 0;
    int limit = (inner_size / kWarpSize) * kWarpSize;
    int outer_id = t_idx / blockDim.x / stride;
    int inner_id = t_idx / blockDim.x % stride;

    /////////////////////////////////////////////
    // init begin
    _Pragma("unroll") for (int i = 0; i < thread_queue; ++i) {
      threadK[i] = init_K;
      threadV[i] = init_V;
    }

    int laneId = GetLaneId();
    int warpId = threadIdx.x / kWarpSize;  // 0,1,2 or 3

    // warp shared memory start address
    warp_K = shared_K + warpId * warp_queue;
    warp_V = shared_V + warpId * warp_queue;

    // landId is 0..31, threadId mod 32
    // warp_quere is nearly above k, 1024 if k=1000
    // each thread takes care of its lane only
    for (int i = laneId; i < warp_queue; i += kWarpSize) {
      warp_K[i] = init_K;
      warp_V[i] = init_V;
    }

    // sync till all threads init done
    __syncwarp();

    // init end
    /////////////////////////////////////////////

    /////////////////////////////////////////////
    // insert begin
    int i = threadIdx.x;
    for (; i < limit; i += threads_per_block) {
      ARG_LEFT_INSERT_THREAD_QUEUE((input[outer_id * inner_size * stride + i * stride + inner_id]), (i));

      // CHECK_AND_MERGE_THREAD_QUEUE() begin
      bool needSort = (num_vals == thread_queue);
      needSort = __any_sync(0xffffffff, needSort);
      if (!needSort) continue;

      MergeWarpQueue<T, S, warp_queue, thread_queue, is_descend>(threadK, threadV, warp_K, warp_V);

      num_vals = 0;
      _Pragma("unroll") for (int i = 0; i < thread_queue; ++i) {
        threadK[i] = init_K;
        threadV[i] = init_V;
      }
      warp_K_top = warp_K[k_minus_1];
      __syncwarp();
    }

    if (i < inner_size) {
      ARG_LEFT_INSERT_THREAD_QUEUE((input[outer_id * inner_size * stride + i * stride + inner_id]), (i));
    }

    // insert end
    /////////////////////////////////////////////

    /////////////////////////////////////////////
    // reduce begin
    MergeWarpQueue<T, S, warp_queue, thread_queue, is_descend>(threadK, threadV, warp_K, warp_V);
    __syncthreads();
    SortBlockWide<kNumWarps, threads_per_block, T, S, warp_queue, is_descend>(shared_K, shared_V);

    // reduce end
    /////////////////////////////////////////////

    // ship data from shared memory to output buffer
    output[outer_id * stride + inner_id] = shared_K[0];
    output_index[outer_id * stride + inner_id] = shared_V[0];
  }
}

template <typename T, typename S>
void ArgmaxWithValue(int outer_size, int inner_size, int stride, const T *input, T *output, S *output_index,
                     cudaStream_t stream) {
  int block_num_limit = outer_size * stride < 1024 ? outer_size * stride : 1024;
  L2_ARGMAX_HELPER(256, 32, 2, true);
}

template <typename T, typename S>
void CalArgmaxWithValue(const T *input, const int bound, const int outerSize, const int innerSize, S *index, T *output,
                        cudaStream_t cuda_stream) {
  ArgmaxWithValue(outerSize, bound, innerSize, input, output, index, cuda_stream);
  return;
}

template void CalArgmaxWithValue<float, int>(const float *input, const int bound_, const int outerSize_,
                                             const int innerSize_, int *index, float *output, cudaStream_t cuda_stream);
template void CalArgmaxWithValue<half, int>(const half *input, const int bound_, const int outerSize_,
                                            const int innerSize_, int *index, half *output, cudaStream_t cuda_stream);
