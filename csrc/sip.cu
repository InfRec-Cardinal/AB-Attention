// Sampling with Inclusion-Probability Weighting (SIP): paper Section 3.2.
//
// Kernel map:
//   1. normalize the per-row masses in shared memory;
//   2. mark certainty (pi == 1) blocks until the residual budget is stable;
//   3. scan once, storing a prefix every four masses;
//   4. locate each draw by bucket search and a walk of at most four masses;
//   5. write sampled KV4 ids and log2(1/pi) weights in the same order.
//
// `sip_kernel` is deliberately a single-block-per-row implementation: the
// shared prefix state makes the without-replacement draw deterministic for a
// given row offset. `sip_cuda` below is only validation, scratch sizing, and
// launch setup.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include "ops.h"

namespace {
constexpr int kThreads = 256;
constexpr int kBucket = 4;

struct SumMax { double sum, maximum; };
struct SumMaxOp {
  __device__ __forceinline__ SumMax operator()(SumMax a, SumMax b) const {
    return {a.sum + b.sum, fmax(a.maximum, b.maximum)};
  }
};

__global__ __launch_bounds__(kThreads) void sip_kernel(
    const float* masses, const float* offsets,
    int32_t* sampled, float* weights, int population, int samples) {
  using Reduce = cub::BlockReduce<SumMax, kThreads>;
  using CountScan = cub::BlockScan<int, kThreads>;
  using MassScan = cub::BlockScan<double, kThreads>;
  union Scratch {
    Reduce::TempStorage reduce;
    CountScan::TempStorage count;
    MassScan::TempStorage mass;
  };
  __shared__ Scratch scratch;
  __shared__ double scale;
  __shared__ int certainty;
  __shared__ int new_certainty;
  __shared__ int last_residual;
  __shared__ int certainty_pass;
  const int buckets = (population + kBucket - 1) / kBucket;
  extern __shared__ double prefix[];
  float* values = reinterpret_cast<float*>(prefix + buckets + 1);
  const int lane = threadIdx.x;
  const int64_t row = blockIdx.x;
  const float* input = masses + row * population;
  int32_t* output = sampled + row * samples;
  float* logs = weights + row * samples;
  for (int index = lane; index < population; index += kThreads)
    values[index] = fmaxf(input[index], 1e-30f);
  if (lane == 0) certainty = 0;
  __syncthreads();
  do {
    double local_sum = 0.;
    double local_max = 0.;
    for (int index = lane; index < population; index += kThreads) {
      if (values[index] > 0.f) {
        const double value = fmax(static_cast<double>(values[index]), 1e-30);
        local_sum += value;
        local_max = fmax(local_max, value);
      }
    }
    const SumMax total = Reduce(scratch.reduce).Reduce({local_sum, local_max}, SumMaxOp{});
    if (lane == 0) {
      scale = (samples - certainty) / total.sum;
      new_certainty = 0;
      // A second full population pass is needed only when at least one
      // unit can become certain.  If max(mass) * scale < 1, all inclusion
      // probabilities are strictly fractional and the marking pass is
      // provably empty.
      certainty_pass = total.maximum * scale >= 1.0;
    }
    __syncthreads();
    if (certainty_pass) {
      int count = 0;
      for (int index = lane; index < population; index += kThreads) {
        if (values[index] > 0.f && fmax(static_cast<double>(values[index]), 1e-30) * scale >= 1.) {
          values[index] = -values[index];
          ++count;
        }
      }
      atomicAdd(&new_certainty, count);
    }
    __syncthreads();
    if (lane == 0) certainty += new_certainty;
    __syncthreads();
  } while (new_certainty != 0);

  // Keep each bucket in one thread; scan chunk totals only once.
  const int chunk = ((buckets + kThreads - 1) / kThreads) * kBucket;
  const int begin = lane * chunk;
  const int end = min(begin + chunk, population);
  if (lane == 0) {
    last_residual = population - 1;
    while (last_residual >= 0 && values[last_residual] <= 0.) --last_residual;
  }
  __syncthreads();

  // Certain blocks precede all residual draws, in increasing index order.
  if (certainty != 0) {
    int count = 0;
    for (int index = begin; index < end; ++index)
      count += values[index] < 0.f;
    int rank = 0;
    CountScan(scratch.count).InclusiveSum(count, rank);
    int slot = rank - count;
    for (int index = begin; index < end; ++index) {
      if (values[index] < 0.f) {
        output[slot] = index;
        logs[slot++] = 0.f;
      }
    }
    // The next scan reuses the same CUB scratch storage.
    __syncthreads();
  }

  double chunk_mass = 0.;
  for (int index = begin; index < end; ++index)
    chunk_mass += values[index] > 0.f ? static_cast<double>(values[index]) * scale : 0.;
  double cumulative = 0.;
  MassScan(scratch.mass).ExclusiveSum(chunk_mass, cumulative);
  const int residuals = samples - certainty;
  if (lane == 0) {
    prefix[buckets] = residuals;
  }
  // prefix[b] is the mass before bucket b. Closing at the last residual
  // ensures that the final threshold cannot escape due to roundoff.
  for (int index = begin; index < end; ++index) {
    if (index % kBucket == 0)
      prefix[index / kBucket] = index > last_residual ? residuals : cumulative;
    cumulative += values[index] > 0.f ? static_cast<double>(values[index]) * scale : 0.;
  }
  __syncthreads();

  for (int slot = lane; slot < residuals; slot += kThreads) {
    const double threshold = static_cast<double>(offsets[row]) + slot;
    int lower = 0, upper = buckets;
    while (lower < upper) {
      const int middle = (lower + upper + 1) / 2;
      if (prefix[middle] <= threshold) lower = middle;
      else upper = middle - 1;
    }
    const int bucket = min(lower, buckets - 1);
    double mass = prefix[bucket];
    int selected = last_residual;
    const int stop = min((bucket + 1) * kBucket, population);
    for (int index = bucket * kBucket; index < stop; ++index) {
      if (values[index] <= 0.f) continue;
      selected = index;
      mass += static_cast<double>(values[index]) * scale;
      if (mass > threshold) break;
    }
    // The bucket search owns the endpoint. At roundoff boundaries, keep
    // its last residual rather than falling through to the end of the row.
    output[certainty + slot] = selected;
    logs[certainty + slot] = static_cast<float>(-log2(fmax(static_cast<double>(input[selected]), 1e-30) * scale));
  }
}
}

void sip_cuda(torch::Tensor scores, torch::Tensor offsets,
              torch::Tensor ids, torch::Tensor logs) {
  TORCH_CHECK(scores.is_cuda() && offsets.device() == scores.device() &&
              ids.device() == scores.device() && logs.device() == scores.device(),
              "all SIP tensors must share a CUDA device");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && offsets.scalar_type() == torch::kFloat32 &&
              ids.scalar_type() == torch::kInt32 && logs.scalar_type() == torch::kFloat32,
              "SIP requires float32 scores/offsets/weights and int32 ids");
  TORCH_CHECK(scores.dim() == 2 && offsets.dim() == 1 && ids.dim() == 2 && ids.sizes() == logs.sizes(),
              "expected scores [rows,N], offsets [rows], ids/weights [rows,samples]");
  TORCH_CHECK(scores.is_contiguous() && offsets.is_contiguous() && ids.is_contiguous() && logs.is_contiguous(),
              "SIP tensors must be contiguous");
  TORCH_CHECK(scores.size(0) == offsets.size(0) && scores.size(0) == ids.size(0), "row counts must agree");
  const int64_t samples = ids.size(1);
  TORCH_CHECK(samples > 0 && samples < scores.size(1), "sample count must be inside the population");
  c10::cuda::CUDAGuard guard(scores.device());
  const size_t buckets = (scores.size(1) + kBucket - 1) / kBucket;
  const size_t bytes = scores.size(1) * sizeof(float) + (buckets + 1) * sizeof(double);
  cudaFuncAttributes attributes;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, sip_kernel));
  int limit;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, scores.get_device()));
  TORCH_CHECK(bytes + attributes.sharedSizeBytes <= static_cast<size_t>(limit), "population exceeds single-block shared memory capacity");
  C10_CUDA_CHECK(cudaFuncSetAttribute(sip_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  // Fill the caller's plan slices directly; no temporary IDs or weight copy.
  sip_kernel<<<scores.size(0), kThreads, bytes, at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), offsets.data_ptr<float>(), ids.data_ptr<int32_t>(),
      logs.data_ptr<float>(), scores.size(1), samples);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
