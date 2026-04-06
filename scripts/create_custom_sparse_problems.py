#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a datapack of custom sparse operator problems that require hand-written CUDA kernels.

These problems mirror the cuSPARSE benchmark problems in interface and tests,
but the build command does NOT link -lcusparse, so solutions must implement
the sparse operations using raw CUDA kernels.

Problems created:
  custom_sparse/0  - SpMV CSR   (Sparse Matrix x Dense Vector, CSR format)
  custom_sparse/1  - SpMM CSR   (Sparse Matrix x Dense Matrix, CSR format)
  custom_sparse/2  - SpVV       (Sparse-Dense dot product)
  custom_sparse/3  - Sparse axpby (scaled sparse-plus-dense)
  custom_sparse/4  - Dense to CSR conversion

Usage:
    python scripts/create_custom_sparse_problems.py
    python scripts/create_custom_sparse_problems.py --release=2026-1 --datapack_dir=data/releases/

Then benchmark with:
    uv run compute_eval generate_samples \\
        --release=2026-1 \\
        --include=sparse \\
        --problems_datapack_dir=data/releases/ \\
        --model=<your-model>
"""

import argparse
from pathlib import Path

from compute_eval.data.data_model import CudaCppProblem, FileSolution, Metadata, ReleaseVersion
from compute_eval.data.data_pack import ProblemDatapack
from compute_eval.data.metrics_data_model import KernelsTimingMode, RegionTimingMode

# ---------------------------------------------------------------------------
# Shared context file: a clean cuda_helpers.h with no cuSPARSE dependency
# ---------------------------------------------------------------------------
CUDA_HELPERS_H = r"""#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cout << "CUDA API failed at line " << __LINE__                      \
                << " with error: " << cudaGetErrorString(status) << " ("       \
                << status << ")" << std::endl;                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#endif // CUDA_HELPERS_H
"""

# ---------------------------------------------------------------------------
# Problem 0 - SpMV CSR
# ---------------------------------------------------------------------------
SPMV_CSR_H = r"""#ifndef SPMV_CSR_H
#define SPMV_CSR_H

#include <vector>

// Compute  y = A * x  where A is a sparse matrix in CSR format.
// A has shape [A_num_rows x A_num_cols], with A_nnz non-zeros.
// hA_csrOffsets: row-pointer array of length A_num_rows+1 (0-based)
// hA_columns   : column indices of each non-zero
// hA_values    : float values of each non-zero
// hX           : input dense vector of length A_num_cols
// hY           : output dense vector of length A_num_rows (overwritten)
void spmv_csr_custom(int A_num_rows, int A_num_cols, int A_nnz,
                     const std::vector<int> &hA_csrOffsets,
                     const std::vector<int> &hA_columns,
                     const std::vector<float> &hA_values,
                     const std::vector<float> &hX,
                     std::vector<float> &hY);

#endif // SPMV_CSR_H
"""

SPMV_CSR_TEST = r"""#include "spmv_csr.h"
#include "cuda_helpers.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <vector>

static void run_benchmark() {
  const int N = 32768;
  const int nnz_per_row = 64;
  const int A_nnz = N * nnz_per_row;

  std::vector<int> hA_csrOffsets(N + 1);
  std::vector<int> hA_columns(A_nnz);
  std::vector<float> hA_values(A_nnz);
  std::vector<float> hX(N, 1.0f);
  std::vector<float> hY(N, 0.0f);

  for (int i = 0; i <= N; i++)
    hA_csrOffsets[i] = i * nnz_per_row;

  srand(42);
  for (int r = 0; r < N; r++) {
    int base = r * nnz_per_row;
    for (int j = 0; j < nnz_per_row; j++) {
      hA_columns[base + j] = (r + j * 511) % N;
      hA_values[base + j] = 1.0f / (1.0f + (float)j);
    }
  }

  const int warmup = 3;
  const int timed = 100;

  for (int i = 0; i < warmup; i++)
    spmv_csr_custom(N, N, A_nnz, hA_csrOffsets, hA_columns, hA_values, hX, hY);
  CHECK_CUDA(cudaDeviceSynchronize());

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; i++)
    spmv_csr_custom(N, N, A_nnz, hA_csrOffsets, hA_columns, hA_values, hX, hY);
  CHECK_CUDA(cudaDeviceSynchronize());
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

  auto spmv_test = []() {
    const int A_num_rows = 4, A_num_cols = 4, A_nnz = 9;
    const std::vector<int> hA_csrOffsets = {0, 3, 4, 7, 9};
    const std::vector<int> hA_columns    = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    const std::vector<float> hA_values   = {1.f, 2.f, 3.f, 4.f, 5.f,
                                            6.f, 7.f, 8.f, 9.f};
    const std::vector<float> hX          = {1.f, 2.f, 3.f, 4.f};
    const std::vector<float> hY_expected = {19.f, 8.f, 51.f, 52.f};
    std::vector<float> hY(A_num_rows, 0.f);

    spmv_csr_custom(A_num_rows, A_num_cols, A_nnz,
                    hA_csrOffsets, hA_columns, hA_values, hX, hY);

    bool ok = true;
    for (int i = 0; i < A_num_rows; i++)
      if (hY[i] != hY_expected[i]) { ok = false; break; }

    if (ok) std::cout << "spmv_csr_custom test PASSED" << std::endl;
    else  { std::cout << "spmv_csr_custom test FAILED" << std::endl;
            std::exit(EXIT_FAILURE); }
  };
  spmv_test();
  return 0;
}
"""

SPMV_CSR_BASELINE = r"""#include "spmv_csr.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>

// One warp (32 threads) handles one row of A.
// Threads in the warp stride over the row's non-zeros, then reduce with
// warp-shuffle to produce the dot product for that row.
__global__ void spmv_csr_warp_kernel(int A_num_rows,
                                     const int  * __restrict__ rowOff,
                                     const int  * __restrict__ colIdx,
                                     const float* __restrict__ vals,
                                     const float* __restrict__ x,
                                           float* __restrict__ y) {
  // warp index within the grid
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane    = threadIdx.x & 31;
  if (warp_id >= A_num_rows) return;

  int row_start = rowOff[warp_id];
  int row_end   = rowOff[warp_id + 1];

  float sum = 0.f;
  for (int j = row_start + lane; j < row_end; j += 32)
    sum += vals[j] * x[colIdx[j]];

  // warp-level reduction
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum,  8);
  sum += __shfl_down_sync(0xffffffff, sum,  4);
  sum += __shfl_down_sync(0xffffffff, sum,  2);
  sum += __shfl_down_sync(0xffffffff, sum,  1);

  if (lane == 0) y[warp_id] = sum;
}

void spmv_csr_custom(int A_num_rows, int A_num_cols, int A_nnz,
                     const std::vector<int>   &hA_csrOffsets,
                     const std::vector<int>   &hA_columns,
                     const std::vector<float> &hA_values,
                     const std::vector<float> &hX,
                           std::vector<float> &hY) {
  int *dRowOff, *dColIdx;
  float *dVals, *dX, *dY;

  CHECK_CUDA(cudaMalloc(&dRowOff, (A_num_rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dColIdx, A_nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dVals,   A_nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dX,   A_num_cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY,   A_num_rows * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dRowOff, hA_csrOffsets.data(),
                        (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dColIdx, hA_columns.data(),
                        A_nnz * sizeof(int),   cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dVals,   hA_values.data(),
                        A_nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(),
                        A_num_cols * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, hY.data(),
                        A_num_rows * sizeof(float), cudaMemcpyHostToDevice));

  // 256 threads / block  ->  8 warps / block  ->  8 rows / block
  const int THREADS = 256;
  int blocks = (A_num_rows + 7) / 8;
  spmv_csr_warp_kernel<<<blocks, THREADS>>>(
      A_num_rows, dRowOff, dColIdx, dVals, dX, dY);

  CHECK_CUDA(cudaMemcpy(hY.data(), dY,
                        A_num_rows * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dRowOff));
  CHECK_CUDA(cudaFree(dColIdx));
  CHECK_CUDA(cudaFree(dVals));
  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dY));
}
"""

# ---------------------------------------------------------------------------
# Problem 1 - SpMM CSR
# ---------------------------------------------------------------------------
SPMM_CSR_H = r"""#ifndef SPMM_CSR_H
#define SPMM_CSR_H

#include <vector>

// Compute  C = alpha * A * B + beta * C
// where A is a sparse [A_num_rows x A_num_cols] matrix in CSR format and
// B, C are dense matrices stored in column-major order.
// B shape: [A_num_cols x B_num_cols]  (leading dim = A_num_cols)
// C shape: [A_num_rows x B_num_cols]  (leading dim = A_num_rows)
void spmm_csr_custom(int A_num_rows, int A_num_cols, int A_nnz,
                     const std::vector<int>   &hA_csrOffsets,
                     const std::vector<int>   &hA_columns,
                     const std::vector<float> &hA_values,
                     int B_num_cols,
                     const std::vector<float> &hB,
                           std::vector<float> &hC,
                     float alpha, float beta);

#endif // SPMM_CSR_H
"""

SPMM_CSR_TEST = r"""#include "spmm_csr.h"
#include "cuda_helpers.h"
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

static void run_benchmark() {
  const int N = 4096;
  const int B_cols = 128;
  const double density = 0.05;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float>  val_dist(0.1f, 1.0f);
  std::uniform_real_distribution<double> coin(0.0, 1.0);

  std::vector<int>   csrOffsets(N + 1, 0);
  std::vector<int>   columns;
  std::vector<float> values;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      if (coin(rng) < density) { columns.push_back(j); values.push_back(val_dist(rng)); }
    csrOffsets[i + 1] = (int)columns.size();
  }
  int nnz = (int)values.size();

  std::vector<float> hB(N * B_cols), hC(N * B_cols, 0.f);
  for (auto &v : hB) v = val_dist(rng);
  float alpha = 1.f, beta = 0.f;

  const int warmup = 3, timed = 100;
  for (int i = 0; i < warmup; i++) {
    std::fill(hC.begin(), hC.end(), 0.f);
    spmm_csr_custom(N, N, nnz, csrOffsets, columns, values, B_cols, hB, hC, alpha, beta);
  }
  for (int i = 0; i < timed; i++) {
    std::fill(hC.begin(), hC.end(), 0.f);
    spmm_csr_custom(N, N, nnz, csrOffsets, columns, values, B_cols, hB, hC, alpha, beta);
  }
}

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

  auto spmm_test = []() {
    int A_num_rows = 4, A_num_cols = 4, A_nnz = 9;
    int B_num_cols = 3, ldb = A_num_cols, ldc = A_num_rows;

    std::vector<int>   hA_csrOffsets = {0, 3, 4, 7, 9};
    std::vector<int>   hA_columns    = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    std::vector<float> hA_values     = {1.f, 2.f, 3.f, 4.f, 5.f,
                                        6.f, 7.f, 8.f, 9.f};
    std::vector<float> hB            = {1.f,  2.f,  3.f,  4.f,
                                        5.f,  6.f,  7.f,  8.f,
                                        9.f, 10.f, 11.f, 12.f};
    std::vector<float> hC(ldc * B_num_cols, 0.f);
    // Column-major expected result
    std::vector<float> hC_result = {19.f,  8.f,  51.f,  52.f,
                                    43.f,  24.f, 123.f, 120.f,
                                    67.f,  40.f, 195.f, 188.f};
    float alpha = 1.f, beta = 0.f;

    spmm_csr_custom(A_num_rows, A_num_cols, A_nnz,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_num_cols, hB, hC, alpha, beta);

    bool ok = true;
    for (int i = 0; i < A_num_rows && ok; i++)
      for (int j = 0; j < B_num_cols && ok; j++)
        if (hC[i + j * ldc] != hC_result[i + j * ldc]) ok = false;

    if (ok) std::cout << "spmm_csr_custom test PASSED" << std::endl;
    else  { std::cout << "spmm_csr_custom test FAILED" << std::endl;
            std::exit(EXIT_FAILURE); }
  };
  spmm_test();
  return 0;
}
"""

SPMM_CSR_BASELINE = r"""#include "spmm_csr.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>

// Grid: (A_num_rows, ceil(B_num_cols/32))
// Block: (32, 1)
// Each thread computes one element C[row, b_col] by dotting the corresponding
// row of A (stored as CSR) with column b_col of B (column-major).
__global__ void spmm_csr_kernel(int A_num_rows, int B_num_cols,
                                 const int  * __restrict__ rowOff,
                                 const int  * __restrict__ colIdx,
                                 const float* __restrict__ vals,
                                 const float* __restrict__ B,
                                       float* __restrict__ C,
                                 float alpha, float beta,
                                 int ldb, int ldc) {
  int row   = blockIdx.x;
  int b_col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= A_num_rows || b_col >= B_num_cols) return;

  float sum = 0.f;
  int row_start = rowOff[row];
  int row_end   = rowOff[row + 1];
  for (int j = row_start; j < row_end; j++)
    sum += vals[j] * B[colIdx[j] + b_col * ldb];  // column-major B

  int out = row + b_col * ldc;                      // column-major C
  C[out] = alpha * sum + beta * C[out];
}

void spmm_csr_custom(int A_num_rows, int A_num_cols, int A_nnz,
                     const std::vector<int>   &hA_csrOffsets,
                     const std::vector<int>   &hA_columns,
                     const std::vector<float> &hA_values,
                     int B_num_cols,
                     const std::vector<float> &hB,
                           std::vector<float> &hC,
                     float alpha, float beta) {
  int ldb = A_num_cols, ldc = A_num_rows;
  int B_size = ldb * B_num_cols, C_size = ldc * B_num_cols;

  int *dRowOff, *dColIdx;
  float *dVals, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dRowOff, (A_num_rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dColIdx, A_nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dVals,   A_nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB,   B_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC,   C_size * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dRowOff, hA_csrOffsets.data(),
                        (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dColIdx, hA_columns.data(),
                        A_nnz * sizeof(int),   cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dVals,   hA_values.data(),
                        A_nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), B_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(), C_size * sizeof(float), cudaMemcpyHostToDevice));

  dim3 grid(A_num_rows, (B_num_cols + 31) / 32);
  dim3 block(32, 1);
  spmm_csr_kernel<<<grid, block>>>(
      A_num_rows, B_num_cols, dRowOff, dColIdx, dVals, dB, dC,
      alpha, beta, ldb, ldc);

  CHECK_CUDA(cudaMemcpy(hC.data(), dC, C_size * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dRowOff)); CHECK_CUDA(cudaFree(dColIdx));
  CHECK_CUDA(cudaFree(dVals));   CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
}
"""

# ---------------------------------------------------------------------------
# Problem 2 - SpVV (Sparse-Dense dot product)
# ---------------------------------------------------------------------------
SPVV_H = r"""#ifndef SPVV_EXAMPLE_H
#define SPVV_EXAMPLE_H

#include <vector>

// Compute the dot product of sparse vector X and dense vector Y.
// X is given in COO format: hX_indices[i] is the index of hX_values[i].
// size  : length of the dense vector Y
// nnz   : number of non-zeros in X
// result: output scalar  sum_i( X[hX_indices[i]] * Y[hX_indices[i]] )
void spvv_custom(int size, int nnz,
                 const std::vector<int>   &hX_indices,
                 const std::vector<float> &hX_values,
                 const std::vector<float> &hY,
                 float &result);

#endif // SPVV_EXAMPLE_H
"""

SPVV_TEST = r"""#include "spvv_example.h"
#include "cuda_helpers.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <vector>

static void run_benchmark() {
  const int size = 10000000;
  const int nnz  = 1000000;

  std::vector<int>   hX_indices(nnz);
  std::vector<float> hX_values(nnz, 1.f);
  std::vector<float> hY(size, 1.f);
  for (int i = 0; i < nnz; i++)
    hX_indices[i] = i * (size / nnz);

  float result;
  for (int i = 0; i < 3; i++)
    spvv_custom(size, nnz, hX_indices, hX_values, hY, result);

  nvtxRangePushA("bench_region");
  for (int i = 0; i < 100; i++)
    spvv_custom(size, nnz, hX_indices, hX_values, hY, result);
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

  auto spvv_test = []() {
    const int size = 8, nnz = 4;
    const std::vector<int>   hX_indices = {0, 3, 4, 7};
    const std::vector<float> hX_values  = {1.f, 2.f, 3.f, 4.f};
    const std::vector<float> hY         = {1.f, 2.f, 3.f, 4.f,
                                           5.f, 6.f, 7.f, 8.f};
    const float expected = 56.f;   // 1*1 + 2*4 + 3*5 + 4*8
    float result;

    spvv_custom(size, nnz, hX_indices, hX_values, hY, result);

    if (result == expected)
      std::cout << "spvv_custom test PASSED" << std::endl;
    else {
      std::cout << "spvv_custom test FAILED: expected " << expected
                << " got " << result << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };
  spvv_test();
  return 0;
}
"""

SPVV_BASELINE = r"""#include "spvv_example.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>

// Each thread multiplies one sparse element x[idx] * y[idx].
// Block-level shared-memory tree reduction accumulates the partial sum;
// atomicAdd merges block sums into the single output scalar.
__global__ void spvv_reduce_kernel(int nnz,
                                   const int  * __restrict__ indices,
                                   const float* __restrict__ xVals,
                                   const float* __restrict__ y,
                                         float* __restrict__ out) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (gid < nnz) ? xVals[gid] * y[indices[gid]] : 0.f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

void spvv_custom(int size, int nnz,
                 const std::vector<int>   &hX_indices,
                 const std::vector<float> &hX_values,
                 const std::vector<float> &hY,
                 float &result) {
  int   *dIdx;
  float *dXVals, *dY, *dOut;

  CHECK_CUDA(cudaMalloc(&dIdx,   nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dXVals, nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY,     size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut,   sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dIdx,   hX_indices.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dXVals, hX_values.data(),  nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, hY.data(), size * sizeof(float),
                        cudaMemcpyHostToDevice));
  float zero = 0.f;
  CHECK_CUDA(cudaMemcpy(dOut, &zero, sizeof(float), cudaMemcpyHostToDevice));

  const int BLOCK = 256;
  int grid = (nnz + BLOCK - 1) / BLOCK;
  spvv_reduce_kernel<<<grid, BLOCK, BLOCK * sizeof(float)>>>(
      nnz, dIdx, dXVals, dY, dOut);

  CHECK_CUDA(cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dIdx));
  CHECK_CUDA(cudaFree(dXVals));
  CHECK_CUDA(cudaFree(dY));
  CHECK_CUDA(cudaFree(dOut));
}
"""

# ---------------------------------------------------------------------------
# Problem 3 - Sparse axpby
# ---------------------------------------------------------------------------
AXPBY_H = r"""#ifndef AXPBY_EXAMPLE_H
#define AXPBY_EXAMPLE_H

#include <vector>

// Compute  hY_result = alpha * X + beta * Y  (sparse axpby).
// X is a sparse vector in COO format: hX_indices[i] / hX_values[i].
// Y is a dense vector of length `size`.
// Only the positions listed in hX_indices are modified by alpha*X;
// all positions receive the beta*Y scaling.
// Result is written to hY_result (same length as Y).
void axpby_custom(int size, int nnz,
                  const std::vector<int>   &hX_indices,
                  const std::vector<float> &hX_values,
                  const std::vector<float> &hY,
                  float alpha, float beta,
                  std::vector<float>       &hY_result);

#endif // AXPBY_EXAMPLE_H
"""

AXPBY_TEST = r"""#include "axpby_example.h"
#include "cuda_helpers.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <vector>

static void run_benchmark() {
  const int size = 2000000;
  const int nnz  = 500000;

  std::vector<int>   hX_indices(nnz);
  std::vector<float> hX_values(nnz);
  std::vector<float> hY(size);
  std::vector<float> hY_result(size);
  float alpha = 2.f, beta = 3.f;

  for (int i = 0; i < nnz; i++) {
    hX_indices[i] = i * (size / nnz);
    hX_values[i]  = (float)(i % 100) * 0.01f;
  }
  for (int i = 0; i < size; i++)
    hY[i] = (float)(i % 200) * 0.005f;

  const int warmup = 3, timed = 100;
  for (int i = 0; i < warmup; i++)
    axpby_custom(size, nnz, hX_indices, hX_values, hY, alpha, beta, hY_result);
  CHECK_CUDA(cudaDeviceSynchronize());

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; i++)
    axpby_custom(size, nnz, hX_indices, hX_values, hY, alpha, beta, hY_result);
  CHECK_CUDA(cudaDeviceSynchronize());
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

  auto test_axpby = []() {
    const int size = 8, nnz = 4;
    const std::vector<int>   hX_indices = {0, 3, 4, 7};
    const std::vector<float> hX_values  = {1.f, 2.f, 3.f, 4.f};
    const std::vector<float> hY         = {1.f, 2.f, 3.f, 4.f,
                                           5.f, 6.f, 7.f, 8.f};
    // expected: beta*Y everywhere; at sparse indices also add alpha*X
    // pos 0: 3*1 + 2*1 = 5   pos 1: 3*2 = 6   pos 2: 3*3 = 9
    // pos 3: 3*4 + 2*2 = 16  pos 4: 3*5 + 2*3 = 21  pos 5: 3*6 = 18
    // pos 6: 3*7 = 21         pos 7: 3*8 + 2*4 = 32
    const std::vector<float> hY_expected = {5.f, 6.f, 9.f, 16.f,
                                            21.f, 18.f, 21.f, 32.f};
    std::vector<float> hY_result(size);
    float alpha = 2.f, beta = 3.f;

    axpby_custom(size, nnz, hX_indices, hX_values, hY, alpha, beta, hY_result);

    bool ok = true;
    for (int i = 0; i < size; i++)
      if (hY_result[i] != hY_expected[i]) { ok = false; break; }

    if (ok) std::cout << "axpby_custom test PASSED" << std::endl;
    else  { std::cout << "axpby_custom test FAILED" << std::endl;
            std::exit(EXIT_FAILURE); }
  };
  test_axpby();
  return 0;
}
"""

AXPBY_BASELINE = r"""#include "axpby_example.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>

// Phase 1: scale the entire dense vector Y by beta  ->  out = beta * Y
__global__ void scale_kernel(int size, float beta,
                              const float* __restrict__ y,
                                    float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) out[i] = beta * y[i];
}

// Phase 2: for each sparse element add  alpha * X[idx]  to the output
__global__ void scatter_add_kernel(int nnz, float alpha,
                                   const int  * __restrict__ indices,
                                   const float* __restrict__ xVals,
                                         float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nnz) atomicAdd(&out[indices[i]], alpha * xVals[i]);
}

void axpby_custom(int size, int nnz,
                  const std::vector<int>   &hX_indices,
                  const std::vector<float> &hX_values,
                  const std::vector<float> &hY,
                  float alpha, float beta,
                  std::vector<float>       &hY_result) {
  int   *dIdx;
  float *dXVals, *dY, *dOut;

  CHECK_CUDA(cudaMalloc(&dIdx,   nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dXVals, nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY,     size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut,   size * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dIdx,   hX_indices.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dXVals, hX_values.data(),  nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, hY.data(), size * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int BLOCK = 256;
  scale_kernel<<<(size + BLOCK - 1) / BLOCK, BLOCK>>>(size, beta, dY, dOut);
  scatter_add_kernel<<<(nnz + BLOCK - 1) / BLOCK, BLOCK>>>(
      nnz, alpha, dIdx, dXVals, dOut);

  CHECK_CUDA(cudaMemcpy(hY_result.data(), dOut, size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dIdx));
  CHECK_CUDA(cudaFree(dXVals));
  CHECK_CUDA(cudaFree(dY));
  CHECK_CUDA(cudaFree(dOut));
}
"""

# ---------------------------------------------------------------------------
# Problem 4 - Dense to CSR conversion
# ---------------------------------------------------------------------------
DENSE_TO_SPARSE_H = r"""#ifndef DENSE_TO_SPARSE_H
#define DENSE_TO_SPARSE_H

#include <vector>

// Convert a dense [num_rows x num_cols] matrix stored in row-major order to
// CSR (Compressed Sparse Row) format.  Zero-valued entries are dropped.
// Outputs are resized by the function.
void dense_to_sparse_csr_custom(int num_rows, int num_cols,
                                const std::vector<float> &h_dense,
                                std::vector<int>         &h_csr_offsets,
                                std::vector<int>         &h_csr_columns,
                                std::vector<float>       &h_csr_values);

#endif // DENSE_TO_SPARSE_H
"""

DENSE_TO_SPARSE_TEST = r"""#include "dense_to_sparse.h"
#include "cuda_helpers.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <vector>

static void run_benchmark() {
  const int num_rows = 4096, num_cols = 4096;
  const int dense_size = num_rows * num_cols;

  std::vector<float> h_dense(dense_size);
  std::srand(42);
  for (int i = 0; i < dense_size; i++)
    h_dense[i] = (std::rand() % 2 == 0) ? 0.f
                                         : (float)(std::rand() % 100 + 1);

  std::vector<int>   h_csr_offsets, h_csr_columns;
  std::vector<float> h_csr_values;

  const int warmup = 3, timed = 100;
  for (int i = 0; i < warmup; i++)
    dense_to_sparse_csr_custom(num_rows, num_cols, h_dense,
                               h_csr_offsets, h_csr_columns, h_csr_values);

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; i++)
    dense_to_sparse_csr_custom(num_rows, num_cols, h_dense,
                               h_csr_offsets, h_csr_columns, h_csr_values);
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

  auto test_dense_to_sparse = []() {
    const int num_rows = 5, num_cols = 4;
    const std::vector<float> h_dense = {
        1.f, 0.f, 2.f, 3.f,
        0.f, 4.f, 0.f, 0.f,
        5.f, 0.f, 6.f, 7.f,
        0.f, 8.f, 0.f, 9.f,
        0.f,10.f,11.f, 0.f};

    std::vector<int>   h_csr_offsets(num_rows + 1, 0);
    std::vector<int>   h_csr_columns(11, 0);
    std::vector<float> h_csr_values(11, 0.f);

    dense_to_sparse_csr_custom(num_rows, num_cols, h_dense,
                               h_csr_offsets, h_csr_columns, h_csr_values);

    const std::vector<int>   ref_offsets = {0, 3, 4, 7, 9, 11};
    const std::vector<int>   ref_cols    = {0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2};
    const std::vector<float> ref_vals    = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
                                            7.f, 8.f, 9.f,10.f,11.f};

    bool ok = true;
    for (int i = 0; i <= num_rows && ok; i++)
      if (h_csr_offsets[i] != ref_offsets[i]) ok = false;
    for (int i = 0; i < 11 && ok; i++)
      if (h_csr_columns[i] != ref_cols[i])  ok = false;
    for (int i = 0; i < 11 && ok; i++)
      if (h_csr_values[i]  != ref_vals[i])  ok = false;

    if (ok) std::cout << "dense_to_sparse_csr_custom test PASSED" << std::endl;
    else  { std::cout << "dense_to_sparse_csr_custom test FAILED" << std::endl;
            std::exit(EXIT_FAILURE); }
  };
  test_dense_to_sparse();
  return 0;
}
"""

DENSE_TO_SPARSE_BASELINE = r"""#include "dense_to_sparse.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <vector>

// Kernel 1: count non-zeros per row (row-major dense matrix)
__global__ void count_nnz_per_row(int num_rows, int num_cols,
                                   const float* __restrict__ dense,
                                         int*   __restrict__ row_nnz) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  int cnt = 0;
  const float *row_ptr = dense + row * num_cols;
  for (int j = 0; j < num_cols; j++)
    if (row_ptr[j] != 0.f) cnt++;
  row_nnz[row] = cnt;
}

// Kernel 2: fill CSR columns and values (each thread handles one row)
__global__ void fill_csr(int num_rows, int num_cols,
                          const float* __restrict__ dense,
                          const int*   __restrict__ csr_offsets,
                                int*   __restrict__ csr_cols,
                                float* __restrict__ csr_vals) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  int ptr = csr_offsets[row];
  const float *row_ptr = dense + row * num_cols;
  for (int j = 0; j < num_cols; j++) {
    float v = row_ptr[j];
    if (v != 0.f) { csr_cols[ptr] = j; csr_vals[ptr] = v; ptr++; }
  }
}

void dense_to_sparse_csr_custom(int num_rows, int num_cols,
                                const std::vector<float> &h_dense,
                                std::vector<int>         &h_csr_offsets,
                                std::vector<int>         &h_csr_columns,
                                std::vector<float>       &h_csr_values) {
  int dense_size = num_rows * num_cols;

  float *d_dense;
  int   *d_row_nnz, *d_csr_offsets;
  CHECK_CUDA(cudaMalloc(&d_dense,       dense_size    * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_row_nnz,     num_rows      * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_csr_offsets, (num_rows + 1)* sizeof(int)));

  CHECK_CUDA(cudaMemcpy(d_dense, h_dense.data(),
                        dense_size * sizeof(float), cudaMemcpyHostToDevice));

  const int BLOCK = 256;
  int grid = (num_rows + BLOCK - 1) / BLOCK;

  // Step 1: count nnz per row
  count_nnz_per_row<<<grid, BLOCK>>>(num_rows, num_cols, d_dense, d_row_nnz);

  // Step 2: exclusive prefix sum -> CSR offsets
  thrust::exclusive_scan(thrust::device,
                         d_row_nnz, d_row_nnz + num_rows + 1,
                         d_csr_offsets);
  // Copy last element (total nnz) from d_row_nnz[num_rows] is 0 after
  // the scan above won't cover it; re-do with a proper range:
  thrust::exclusive_scan(thrust::device,
                         d_row_nnz, d_row_nnz + num_rows,
                         d_csr_offsets);
  // Set d_csr_offsets[num_rows] = total nnz
  int total_nnz_host;
  {
    int last_row_nnz, last_off;
    CHECK_CUDA(cudaMemcpy(&last_row_nnz, d_row_nnz + num_rows - 1,
                          sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&last_off,     d_csr_offsets + num_rows - 1,
                          sizeof(int), cudaMemcpyDeviceToHost));
    total_nnz_host = last_off + last_row_nnz;
    CHECK_CUDA(cudaMemcpy(d_csr_offsets + num_rows, &total_nnz_host,
                          sizeof(int), cudaMemcpyHostToDevice));
  }

  // Step 3: allocate and fill CSR arrays
  int   *d_csr_cols;
  float *d_csr_vals;
  CHECK_CUDA(cudaMalloc(&d_csr_cols, total_nnz_host * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_csr_vals, total_nnz_host * sizeof(float)));

  fill_csr<<<grid, BLOCK>>>(num_rows, num_cols, d_dense,
                              d_csr_offsets, d_csr_cols, d_csr_vals);

  // Copy results back to host
  h_csr_offsets.resize(num_rows + 1);
  h_csr_columns.resize(total_nnz_host);
  h_csr_values.resize(total_nnz_host);

  CHECK_CUDA(cudaMemcpy(h_csr_offsets.data(), d_csr_offsets,
                        (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_csr_columns.data(), d_csr_cols,
                        total_nnz_host * sizeof(int),   cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_csr_values.data(),  d_csr_vals,
                        total_nnz_host * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_dense));
  CHECK_CUDA(cudaFree(d_row_nnz));
  CHECK_CUDA(cudaFree(d_csr_offsets));
  CHECK_CUDA(cudaFree(d_csr_cols));
  CHECK_CUDA(cudaFree(d_csr_vals));
}
"""

# ---------------------------------------------------------------------------
# Build the problem objects
# ---------------------------------------------------------------------------

def make_problem(task_id, prompt, context_files, test_files, baseline_files,
                 build_cmd, timing_mode, tags, release):
    return CudaCppProblem(
        type="cuda_cpp",
        task_id=task_id,
        date="2026-04-06",
        prompt=prompt,
        metadata=Metadata(
            difficulty="medium",
            tags=tags,
            releases=[release],
        ),
        group="sparse",
        context_files=[{"path": p, "content": c} for p, c in context_files],
        test_files=[{"path": p, "content": c} for p, c in test_files],
        build_command=build_cmd,
        test_command="./test.out",
        benchmark_command="./test.out --perf",
        timing_mode=timing_mode,
        min_cuda_toolkit="12.0",
        compute_capability="8.0",
        requires_datacenter_gpu=False,
        timeout_seconds=120.0,
        baseline_solution=FileSolution(
            type="file",
            task_id=task_id,
            files=[{"path": p, "content": c} for p, c in baseline_files],
        ),
    )


BUILD_NO_CUSPARSE = "nvcc -I include -o test.out solution.cu test/*.cu -arch=native"
BUILD_THRUST      = "nvcc -I include -o test.out solution.cu test/*.cu -arch=native"

RELEASE = ReleaseVersion.V2026_1


def build_problems():
    return [
        # ------------------------------------------------------------------
        # 0: SpMV CSR
        # ------------------------------------------------------------------
        make_problem(
            task_id="custom_sparse/0",
            prompt="""Write a function `void spmv_csr_custom(...)` that computes Sparse Matrix x Dense Vector multiplication (SpMV) in CSR format using custom CUDA kernels - **do not use cuSPARSE or any other sparse library**.

The function must:
- Accept a sparse matrix A in CSR format (row offsets, column indices, float values) and a dense vector X on the host.
- Allocate all device memory internally, transfer the inputs to the GPU, launch your CUDA kernel(s), copy the result back to the host, and free all device memory before returning.
- Write the result into the pre-allocated output vector `hY` (length A_num_rows).

Your implementation should be in `solution.cu` and implement the function declared in `include/spmv_csr.h`:
```cpp
void spmv_csr_custom(int A_num_rows, int A_num_cols, int A_nnz,
                     const std::vector<int>   &hA_csrOffsets,
                     const std::vector<int>   &hA_columns,
                     const std::vector<float> &hA_values,
                     const std::vector<float> &hX,
                           std::vector<float> &hY);
```

Use the `CHECK_CUDA` macro from `include/cuda_helpers.h` for error handling.
Include `<cuda_runtime.h>` for device functions (e.g. `__shfl_down_sync`).

Hint: a warp-per-row approach (32 threads per row, warp-shuffle reduction) is a good starting point.""",
            context_files=[
                ("include/spmv_csr.h",     SPMV_CSR_H),
                ("include/cuda_helpers.h", CUDA_HELPERS_H),
            ],
            test_files=[("test/test_main.cu", SPMV_CSR_TEST)],
            baseline_files=[("solution.cu", SPMV_CSR_BASELINE)],
            build_cmd=BUILD_NO_CUSPARSE,
            timing_mode=RegionTimingMode(include=["bench_region"], time_type="kernel"),
            tags=["custom-kernel", "sparse-matrix", "spmv", "csr", "perf-sensitive"],
            release=RELEASE,
        ),

        # ------------------------------------------------------------------
        # 1: SpMM CSR
        # ------------------------------------------------------------------
        make_problem(
            task_id="custom_sparse/1",
            prompt="""Write a function `void spmm_csr_custom(...)` that performs Sparse Matrix x Dense Matrix multiplication (SpMM) in CSR format using custom CUDA kernels - **do not use cuSPARSE or any other sparse library**.

The function must:
- Accept a sparse matrix A in CSR format and dense matrices B and C on the host (both stored in **column-major** order).
- Compute  C = alpha * A * B + beta * C.
- Allocate all device memory internally, transfer inputs, launch your kernel(s), copy C back to host, and free all device memory.

Your implementation should be in `solution.cu` and implement the function declared in `include/spmm_csr.h`:
```cpp
void spmm_csr_custom(int A_num_rows, int A_num_cols, int A_nnz,
                     const std::vector<int>   &hA_csrOffsets,
                     const std::vector<int>   &hA_columns,
                     const std::vector<float> &hA_values,
                     int B_num_cols,
                     const std::vector<float> &hB,
                           std::vector<float> &hC,
                     float alpha, float beta);
```
- B is column-major with leading dimension A_num_cols.
- C is column-major with leading dimension A_num_rows.

Use the `CHECK_CUDA` macro from `include/cuda_helpers.h` for error handling.""",
            context_files=[
                ("include/spmm_csr.h",     SPMM_CSR_H),
                ("include/cuda_helpers.h", CUDA_HELPERS_H),
            ],
            test_files=[("test/test_main.cu", SPMM_CSR_TEST)],
            baseline_files=[("solution.cu", SPMM_CSR_BASELINE)],
            build_cmd=BUILD_NO_CUSPARSE,
            timing_mode=KernelsTimingMode(),
            tags=["custom-kernel", "sparse-matrix", "spmm", "csr", "perf-sensitive"],
            release=RELEASE,
        ),

        # ------------------------------------------------------------------
        # 2: SpVV
        # ------------------------------------------------------------------
        make_problem(
            task_id="custom_sparse/2",
            prompt="""Write a function `void spvv_custom(...)` that computes the dot product between a sparse vector X and a dense vector Y using custom CUDA kernels - **do not use cuSPARSE or any other sparse library**.

The function must:
- Accept sparse vector X in COO format (index array + value array) and dense vector Y on the host.
- Compute  result = sum_i( X_values[i] * Y[X_indices[i]] ).
- Allocate device memory internally, transfer inputs, launch your kernel(s), copy the scalar result back, and free all device memory.

Your implementation should be in `solution.cu` and implement the function declared in `include/spvv_example.h`:
```cpp
void spvv_custom(int size, int nnz,
                 const std::vector<int>   &hX_indices,
                 const std::vector<float> &hX_values,
                 const std::vector<float> &hY,
                 float &result);
```

Use the `CHECK_CUDA` macro from `include/cuda_helpers.h` for error handling.

Hint: compute `xVals[i] * Y[idx[i]]` per thread, reduce within each block using shared memory, then merge block sums with `atomicAdd`.""",
            context_files=[
                ("include/spvv_example.h", SPVV_H),
                ("include/cuda_helpers.h", CUDA_HELPERS_H),
            ],
            test_files=[("test/test_main.cu", SPVV_TEST)],
            baseline_files=[("solution.cu", SPVV_BASELINE)],
            build_cmd=BUILD_NO_CUSPARSE,
            timing_mode=RegionTimingMode(include=["bench_region"], time_type="kernel"),
            tags=["custom-kernel", "sparse-vector", "reduction", "spvv", "perf-sensitive"],
            release=RELEASE,
        ),

        # ------------------------------------------------------------------
        # 3: Sparse axpby
        # ------------------------------------------------------------------
        make_problem(
            task_id="custom_sparse/3",
            prompt="""Write a function `void axpby_custom(...)` that computes a sparse axpby operation using custom CUDA kernels - **do not use cuSPARSE or any other sparse library**.

The operation is:  hY_result[i] = alpha * X[i] + beta * Y[i]
where X is a sparse vector in COO format (only `nnz` elements are non-zero) and Y is a dense vector of length `size`.

The function must:
- Scale the full dense vector Y by beta.
- Add alpha * X to the appropriate positions (scatter-add).
- Handle all device memory allocation, data transfers, kernel launches, result copy-back, and cleanup internally.

Your implementation should be in `solution.cu` and implement the function declared in `include/axpby_example.h`:
```cpp
void axpby_custom(int size, int nnz,
                  const std::vector<int>   &hX_indices,
                  const std::vector<float> &hX_values,
                  const std::vector<float> &hY,
                  float alpha, float beta,
                  std::vector<float>       &hY_result);
```

Use the `CHECK_CUDA` macro from `include/cuda_helpers.h` for error handling.""",
            context_files=[
                ("include/axpby_example.h", AXPBY_H),
                ("include/cuda_helpers.h",  CUDA_HELPERS_H),
            ],
            test_files=[("test/test_main.cu", AXPBY_TEST)],
            baseline_files=[("solution.cu", AXPBY_BASELINE)],
            build_cmd=BUILD_NO_CUSPARSE,
            timing_mode=RegionTimingMode(include=["bench_region"], time_type="kernel"),
            tags=["custom-kernel", "sparse-vector", "axpby", "scatter", "perf-sensitive"],
            release=RELEASE,
        ),

        # ------------------------------------------------------------------
        # 4: Dense to CSR conversion
        # ------------------------------------------------------------------
        make_problem(
            task_id="custom_sparse/4",
            prompt="""Write a function `void dense_to_sparse_csr_custom(...)` that converts a row-major dense matrix to CSR (Compressed Sparse Row) format on the GPU using custom CUDA kernels - **do not use cuSPARSE or any other sparse library**.

The function must:
1. Count the number of non-zero elements in each row.
2. Compute the CSR row-pointer array (exclusive prefix sum of per-row counts).
3. Fill the column-index and value arrays from the non-zero positions.
4. Handle all device memory allocation, data transfers, kernel launches, and cleanup.
5. Resize and populate the output vectors before returning.

Your implementation should be in `solution.cu` and implement the function declared in `include/dense_to_sparse.h`:
```cpp
void dense_to_sparse_csr_custom(int num_rows, int num_cols,
                                const std::vector<float> &h_dense,
                                std::vector<int>         &h_csr_offsets,
                                std::vector<int>         &h_csr_columns,
                                std::vector<float>       &h_csr_values);
```

The dense matrix is stored in **row-major** order (`h_dense[row * num_cols + col]`).  Zero-valued entries must be excluded from the output.

Use the `CHECK_CUDA` macro from `include/cuda_helpers.h` for error handling.
You may use Thrust (e.g. `thrust::exclusive_scan`) for the prefix sum step.""",
            context_files=[
                ("include/dense_to_sparse.h", DENSE_TO_SPARSE_H),
                ("include/cuda_helpers.h",     CUDA_HELPERS_H),
            ],
            test_files=[("test/test_main.cu", DENSE_TO_SPARSE_TEST)],
            baseline_files=[("solution.cu", DENSE_TO_SPARSE_BASELINE)],
            build_cmd=BUILD_THRUST,
            timing_mode=RegionTimingMode(include=["bench_region"], time_type="kernel"),
            tags=["custom-kernel", "sparse-format", "csr", "dense-to-sparse", "thrust", "perf-sensitive"],
            release=RELEASE,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_custom_sparse_datapack(release: str, datapack_dir: str) -> None:
    datapack_dir_path = Path(datapack_dir)
    release_version = ReleaseVersion(release)
    output_path = datapack_dir_path / f"{release}-custom-sparse-problems.tar.gz"

    problems = build_problems()

    ProblemDatapack.create(
        file_path=output_path,
        items=problems,
        release=release_version,
        description=(
            f"Custom sparse operator problems ({release}): "
            "SpMV/SpMM/SpVV/axpby/dense-to-CSR implemented with hand-written CUDA kernels, "
            "no cuSPARSE dependency."
        ),
    )

    print(f"Created custom sparse datapack: {output_path}")
    print(f"  Problems : {len(problems)}")
    for p in problems:
        print(f"    {p.task_id}")
    print()
    print("Next steps:")
    print(f"  # Generate solutions")
    print(f"  uv run compute_eval generate_samples \\")
    print(f"    --release={release} \\")
    print(f"    --include=sparse \\")
    print(f"    --problems_datapack={output_path} \\")
    print(f"    --model=<your-model>")
    print()
    print(f"  # Evaluate correctness")
    print(f"  uv run compute_eval evaluate_functional_correctness \\")
    print(f"    --release={release} \\")
    print(f"    --solutions_datapack=<solutions.tar.gz> \\")
    print(f"    --problems_datapack={output_path} \\")
    print(f"    --mode=docker")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release",      default="2026-1",
                        help="Release version (default: 2026-1)")
    parser.add_argument("--datapack_dir", default="data/releases/",
                        help="Output directory for the datapack")
    args = parser.parse_args()
    create_custom_sparse_datapack(args.release, args.datapack_dir)


if __name__ == "__main__":
    main()
