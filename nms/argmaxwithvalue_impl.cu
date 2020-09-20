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

template <typename T, typename S, int threads_per_block>
__global__ void L2ArgMax(int outer_size, int inner_size, int stride, const T *input, T *output, S *output_index) {
  constexpr int kNumWarps = threads_per_block / kWarpSize;
  const int threads_per_warp = 32;
  const T init_K = static_cast<T>(-9999);
  constexpr S init_V = static_cast<S>(-1);

  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < blockDim.x * outer_size * stride;
       t_idx += blockDim.x * gridDim.x) {
    __shared__ T shared_K[kNumWarps];
    __shared__ S shared_V[kNumWarps];

    T warp_K_top = init_K;
    int outer_id = t_idx / blockDim.x / stride;
    int inner_id = t_idx / blockDim.x % stride;

    T threadK = init_K;
    S threadV = init_V;

    int laneId = GetLaneId();
    int warpId = threadIdx.x / kWarpSize;  // 0,1,2 or 3

    // sync till all threads init done
    __syncwarp();

    int i = threadIdx.x;
    for (; i < inner_size; i += threads_per_block) {
      auto &k = input[outer_id * inner_size * stride + i * stride + inner_id];
      auto &v = i;
      if (Cmp<T>::gt(k, warp_K_top)) {
        {
          threadK = k;
          threadV = v;
          warp_K_top = k;
        }
      }
    }
    __syncwarp();

    for (int offset = threads_per_warp / 2; offset > 0; offset /= 2) {
      T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
      S other_V = __shfl_down_sync(0xffffffff, threadV, offset);

      bool small_compare_descend = Cmp<T>::lt(threadK, other_K);
      ConditionAssign(small_compare_descend, &threadK, other_K);
      ConditionAssign(small_compare_descend, &threadV, other_V);
    }
    shared_K[warpId] = threadK;
    shared_V[warpId] = threadV;
    __syncthreads();

    _Pragma("unroll") for (int offset = kNumWarps / 2; offset > 0; offset /= 2) {
      int pos = threadIdx.x;
      if ( pos >= (kNumWarps/offset)) break; 
      L2CompareAndSwap<T, S, false>(shared_K, shared_V, pos, pos + offset);
    }
    __syncwarp();

    output[outer_id * stride + inner_id] = shared_K[0];
    output_index[outer_id * stride + inner_id] = shared_V[0];
  }
}

template <typename T, typename S>
void ArgmaxWithValue(int outer_size, int inner_size, int stride, const T *input, T *output, S *output_index,
                     cudaStream_t stream) {
  int block_num_limit = outer_size * stride < 1024 ? outer_size * stride : 1024;
  L2ArgMax<T, S, 256><<<GET_BLOCKS((256 * block_num_limit)), 256, 0, stream>>>(outer_size, inner_size, stride, input,
                                                                               output, output_index);
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
