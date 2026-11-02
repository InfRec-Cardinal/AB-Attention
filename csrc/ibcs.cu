// Intra-Block Clustering Selector (IBCS): paper Section 3.1.
//
// Kernel map (the wrapper launches these in order):
//   1. cluster_queries_kernel: one 128-token Q block -> eight centroids;
//   2. pool_keys_kernel:      one KV4 block -> two key representatives;
//   3. ibcs_lse_kernel:       centroid/key scores -> per-centroid LSE;
//   4. ibcs_scores_kernel:    centroid populations + LSE -> one mass per KV4.
//
// The public `ibcs_cuda` wrapper below owns shapes, temporary buffers, and
// launch order. The kernels above only implement the corresponding stages.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <torch/extension.h>

#include "ops.h"
#include "flashinfer/permuted_smem.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

using Bf16 = __nv_bfloat16;
using SwizzledSmem = flashinfer::smem_t<flashinfer::SwizzleMode::k128B>;
using MmaMode = flashinfer::mma::MMAMode;

constexpr int kHeadDim = 128;
constexpr int kQueryBlockSize = 128;
constexpr int kQueryCentroids = 8;
constexpr int kSamplingBlockSize = 4;
constexpr int kKeyRepresentatives = 2;
// Each pooled representative covers two original tokens, so its log-count is
// one. Keep the shift in the scoring path instead of allocating a buffer.
constexpr float kLogRepresentativeSize = 1.0f;
constexpr int kKeyBlocksPerTile = 8;
constexpr int kMmaTile = 16;
constexpr int kMmaTilesHeadDim = kHeadDim / kMmaTile;
constexpr int kFragmentElements = 8;
constexpr int kBf16PerVector = sizeof(uint4) / sizeof(Bf16);
constexpr int kVectorsPerRow = kHeadDim / kBf16PerVector;
constexpr int kStageElements = kMmaTile * kHeadDim;
constexpr size_t kStageBytes = kStageElements * sizeof(Bf16);
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kScoreWarps = kQueryCentroids;
constexpr int kScoreThreads = kScoreWarps * 32;
constexpr int kQBlocksPerGroup = 2;
constexpr size_t kSharedBytes = 2 * kStageBytes + kWarps * kStageBytes;

static_assert(kQueryCentroids * kQBlocksPerGroup == kMmaTile);
static_assert(kKeyRepresentatives * kKeyBlocksPerTile == kMmaTile);

using ArgMaxValue = cub::KeyValuePair<int, float>;
using QueryBlockReduce = cub::BlockReduce<ArgMaxValue, kThreads>;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, mask);
    }
    return value;
}

// ---- Shared math ----------------------------------------------------------

__device__ __forceinline__ void mma_qk_tile(
    SwizzledSmem k_smem,
    uint32_t (&q_frag)[kMmaTilesHeadDim][4],
    float (&score_frag)[kFragmentElements],
    int lane) {
#pragma unroll
    for (int mma_k = 0; mma_k < kMmaTilesHeadDim; ++mma_k) {
        uint32_t k_frag[4];
        const int row = 8 * (lane / 16) + lane % 8;
        const int vector_col = mma_k * 2 + (lane % 16) / 8;
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(row, vector_col);
        k_smem.ldmatrix_m8n8x4(smem_offset, k_frag);
        if (mma_k == 0) {
            flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<Bf16, MmaMode::kInit>(
                score_frag, q_frag[mma_k], k_frag);
        } else {
            flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<Bf16>(
                score_frag, q_frag[mma_k], k_frag);
        }
    }
}

// Block-wide argmax over one value per thread. CUB owns the warp/block
// reduction; CUB ArgMax breaks ties by the smallest token index.
// The return value is valid only in thread 0.
__device__ __forceinline__ int block_argmax(
    float value, int index, QueryBlockReduce::TempStorage& storage) {
    const ArgMaxValue best =
        QueryBlockReduce(storage).Reduce(ArgMaxValue{index, value}, cub::ArgMax{});
    return best.key;
}

// ---- Query clustering ------------------------------------------------------

__global__ __launch_bounds__(kHeadDim)
void cluster_queries_kernel(
    const Bf16* __restrict__ q,
    Bf16* __restrict__ packed_q,
    half* __restrict__ populations,
    int64_t seq_len,
    int64_t q_blocks) {
    __shared__ Bf16 queries[kQueryBlockSize * kHeadDim];
    __shared__ float mean[kHeadDim];
    __shared__ float min_distance[kQueryBlockSize];
    __shared__ int assignments[kQueryBlockSize];
    __shared__ int center;
    __shared__ int counts[kQueryCentroids];
    __shared__ QueryBlockReduce::TempStorage argmax_storage;

    const int dim = threadIdx.x;
    const int warp = dim / 32;
    const int lane = dim & 31;
    const int64_t q_block = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t token_base = q_block * kQueryBlockSize;
    const int valid_queries = static_cast<int>(
        min(static_cast<int64_t>(kQueryBlockSize), seq_len - token_base));

    float sum = 0.0f;
#pragma unroll
    for (int query = 0; query < kQueryBlockSize; ++query) {
        Bf16 value = __float2bfloat16_rn(0.0f);
        if (query < valid_queries) {
            value = q[(bh * seq_len + token_base + query) * kHeadDim + dim];
            sum += __bfloat162float(value);
        }
        queries[query * kHeadDim + dim] = value;
    }
    mean[dim] = sum / static_cast<float>(valid_queries);
    __syncthreads();

    for (int query = warp; query < valid_queries; query += kWarps) {
        float distance = 0.0f;
        for (int local_dim = lane; local_dim < kHeadDim; local_dim += 32) {
            const float delta =
                __bfloat162float(queries[query * kHeadDim + local_dim]) -
                mean[local_dim];
            distance += delta * delta;
        }
        distance = warp_sum(distance);
        if (lane == 0) {
            min_distance[query] = distance;
        }
    }
    __syncthreads();

    // One query per thread: the farthest-first scan is a block reduction, not a
    // serial walk by thread 0 with the other 127 threads waiting on it.
    const int first_center = block_argmax(
        dim < valid_queries ? min_distance[dim] : -INFINITY, dim,
        argmax_storage);
    if (dim == 0) center = first_center;
    __syncthreads();
    if (dim < valid_queries) {
        min_distance[dim] = INFINITY;
        assignments[dim] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int center_slot = 0; center_slot < kQueryCentroids; ++center_slot) {
        for (int query = warp; query < valid_queries; query += kWarps) {
            float distance = 0.0f;
            for (int local_dim = lane; local_dim < kHeadDim; local_dim += 32) {
                const float delta =
                    __bfloat162float(queries[query * kHeadDim + local_dim]) -
                    __bfloat162float(queries[center * kHeadDim + local_dim]);
                distance += delta * delta;
            }
            distance = warp_sum(distance);
            if (lane == 0 && distance < min_distance[query]) {
                min_distance[query] = distance;
                assignments[query] = center_slot;
            }
        }
        __syncthreads();
        // Assign to all eight centers; choose a successor only when needed.
        // center_slot is uniform across the block, including its barriers.
        if (center_slot + 1 < kQueryCentroids) {
            const int next_center = block_argmax(
                dim < valid_queries ? min_distance[dim] : -INFINITY, dim,
                argmax_storage);
            if (dim == 0) center = next_center;
            __syncthreads();
        }
    }

    if (dim < kQueryCentroids) {
        counts[dim] = 0;
    }
    __syncthreads();
    if (dim < valid_queries) {
        atomicAdd(&counts[assignments[dim]], 1);
    }
    __syncthreads();
    if (dim < kQueryCentroids) {
        populations[(bh * q_blocks + q_block) * kQueryCentroids + dim] =
            __float2half_rn(static_cast<float>(counts[dim]));
    }
    __syncthreads();

    float centroid_sum[kQueryCentroids];
#pragma unroll
    for (int cluster = 0; cluster < kQueryCentroids; ++cluster) {
        centroid_sum[cluster] = 0.0f;
    }
    // Indexing centroid_sum by the assignment would spill it to local memory,
    // so select over the eight clusters instead and keep the sums in registers.
    for (int query = 0; query < valid_queries; ++query) {
        const float value = __bfloat162float(queries[query * kHeadDim + dim]);
        const int cluster = assignments[query];
#pragma unroll
        for (int slot = 0; slot < kQueryCentroids; ++slot) {
            centroid_sum[slot] += (slot == cluster) ? value : 0.0f;
        }
    }
    const int64_t output_base =
        (bh * q_blocks + q_block) * kQueryCentroids * kHeadDim;
#pragma unroll
    for (int cluster = 0; cluster < kQueryCentroids; ++cluster) {
        packed_q[output_base + cluster * kHeadDim + dim] =
            counts[cluster] > 0
            ? __float2bfloat16_rn(
                  centroid_sum[cluster] / static_cast<float>(counts[cluster]))
            : __float2bfloat16_rn(0.0f);
    }
}


// ---- Key pooling -----------------------------------------------------------

// Pool each KV4 block into two representatives (means of token pairs).
__global__ __launch_bounds__(kHeadDim)
void pool_keys_kernel(
    const Bf16* __restrict__ k,
    Bf16* __restrict__ packed_k,
    int64_t fine_blocks) {
    const int dim = threadIdx.x;
    const int64_t fine_block = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t source_base =
        (bh * fine_blocks + fine_block) * kSamplingBlockSize * kHeadDim;
    const int64_t output_base =
        (bh * fine_blocks + fine_block) * kKeyRepresentatives * kHeadDim;
#pragma unroll
    for (int representative = 0; representative < kKeyRepresentatives;
         ++representative) {
        const int first = representative * 2;
        const float value = 0.5f * (
            __bfloat162float(k[source_base + first * kHeadDim + dim]) +
            __bfloat162float(k[source_base + (first + 1) * kHeadDim + dim]));
        packed_k[output_base + representative * kHeadDim + dim] =
            __float2bfloat16_rn(value);
    }
}

// ---- Tiled centroid/key scoring -------------------------------------------

__device__ __forceinline__ void load_q_centroid_pair(
    const Bf16* __restrict__ packed_q,
    SwizzledSmem q_smem,
    int lane,
    int64_t bh,
    int64_t q_block_base,
    int64_t q_blocks,
    float qk_scale_log2) {
    constexpr int kTileVectors = kMmaTile * kVectorsPerRow;
    for (int vector_idx = lane; vector_idx < kTileVectors; vector_idx += 32) {
        const int row = vector_idx / kVectorsPerRow;
        const int vector_col = vector_idx - row * kVectorsPerRow;
        const int local_q_block = row / kQueryCentroids;
        const int centroid = row - local_q_block * kQueryCentroids;
        const int64_t q_block = q_block_base + local_q_block;
        uint4 packed = make_uint4(0, 0, 0, 0);
        Bf16* values = reinterpret_cast<Bf16*>(&packed);
        if (q_block < q_blocks) {
            const int64_t source_base =
                ((bh * q_blocks + q_block) * kQueryCentroids + centroid) *
                kHeadDim + vector_col * kBf16PerVector;
#pragma unroll
            for (int elem = 0; elem < kBf16PerVector; ++elem) {
                values[elem] = __float2bfloat16_rn(
                    __bfloat162float(packed_q[source_base + elem]) *
                    qk_scale_log2);
            }
        }
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(row, vector_col);
        q_smem.base[smem_offset] = packed;
    }
}

__device__ __forceinline__ void stage_pooled_keys_async(
    const Bf16* __restrict__ packed_k,
    SwizzledSmem k_smem,
    int thread,
    int64_t bh,
    int fine_block_base,
    int fine_blocks) {
    constexpr int kTileVectors = kMmaTile * kVectorsPerRow;
    for (int vector_idx = thread; vector_idx < kTileVectors;
         vector_idx += kThreads) {
        const int row = vector_idx / kVectorsPerRow;
        const int vector_col = vector_idx - row * kVectorsPerRow;
        const int local_block = row / kKeyRepresentatives;
        const int representative = row - local_block * kKeyRepresentatives;
        const int fine_block = min(fine_block_base + local_block, fine_blocks - 1);
        const int64_t source_base =
            ((bh * fine_blocks + fine_block) * kKeyRepresentatives + representative) *
            kHeadDim + vector_col * kBf16PerVector;
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(row, vector_col);
        k_smem.load_128b_async(smem_offset, packed_k + source_base);
    }
}

// Fold one log-sum-exp term into a running (max, denominator) pair.
__device__ __forceinline__ void accumulate_lse(
    float& maximum, float& denominator, float value) {
    const float next_max = fmaxf(maximum, value);
    denominator = denominator * exp2f(maximum - next_max) +
        exp2f(value - next_max);
    maximum = next_max;
}

// log2-sum-exp of two terms, in base 2. The floor keeps a pair of very negative
// scores from producing a zero denominator.
__device__ __forceinline__ float pair_lse(float left, float right) {
    const float maximum = fmaxf(left, right);
    return maximum + log2f(
        fmaxf(exp2f(left - maximum) + exp2f(right - maximum), 1.0e-20f));
}

// Store one block's log2 softmax normalizer and fold it into the running
// normalizer for the query block that owns it.
__device__ __forceinline__ void store_block_lse(
    half* __restrict__ output,
    int64_t row_base,
    int block,
    int fine_blocks,
    half stored,
    float& maximum,
    float& denominator) {
    if (block < fine_blocks) {
        output[row_base + block] = stored;
        accumulate_lse(maximum, denominator, __half2float(stored));
    }
}

// Merge two (max, denominator) normalizer pairs covering disjoint key blocks.
__device__ __forceinline__ void combine_lse(
    float& maximum, float& denominator, float other_max, float other_den) {
    const float next = fmaxf(maximum, other_max);
    denominator = denominator * exp2f(maximum - next) +
        other_den * exp2f(other_max - next);
    maximum = next;
}

__global__ __launch_bounds__(kThreads)
void ibcs_lse_kernel(
    const Bf16* __restrict__ packed_q,
    const Bf16* __restrict__ packed_k,
    half* __restrict__ output,
    float* __restrict__ row_max,
    float* __restrict__ row_den,
    int64_t q_blocks,
    int fine_blocks,
    float qk_scale_log2) {
    extern __shared__ __align__(32) unsigned char smem_raw[];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int64_t bh = blockIdx.y;
    const int64_t q_group = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
    const int64_t q_block_base = q_group * kQBlocksPerGroup;
    unsigned char* warp_q_smem =
        smem_raw + 2 * kStageBytes + warp * kStageBytes;
    SwizzledSmem q_stage(warp_q_smem);
    SwizzledSmem k_stage0(smem_raw);
    SwizzledSmem k_stage1(smem_raw + kStageBytes);

    load_q_centroid_pair(
        packed_q,
        q_stage,
        lane,
        bh,
        q_block_base,
        q_blocks,
        qk_scale_log2);
    __syncwarp();
    uint32_t q_frag[kMmaTilesHeadDim][4];
#pragma unroll
    for (int mma_k = 0; mma_k < kMmaTilesHeadDim; ++mma_k) {
        const int q_row = lane % 16;
        const int vector_col = mma_k * 2 + lane / 16;
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(q_row, vector_col);
        q_stage.ldmatrix_m8n8x4(smem_offset, q_frag[mma_k]);
    }
    __syncwarp();

    stage_pooled_keys_async(
        packed_k, k_stage0, threadIdx.x, bh, 0, fine_blocks);
    flashinfer::cp_async::commit_group();
    const int tile_count =
        (fine_blocks + kKeyBlocksPerTile - 1) / kKeyBlocksPerTile;
    // Each lane owns centroid lane/4 for both query blocks of the group and
    // visits every key block of that centroid, so the softmax normalizer is
    // accumulated here instead of in a second pass over the LSE array.
    // Empty lanes need a finite maximum so merging them never forms -inf - -inf.
    // FP16 LSEs are far above -FLT_MAX; empty lanes still contribute exactly zero.
    float maximum[kQBlocksPerGroup] = {-FLT_MAX, -FLT_MAX};
    float denominator[kQBlocksPerGroup] = {0.0f, 0.0f};
    for (int tile = 0; tile < tile_count; ++tile) {
        const bool use_stage0 = (tile & 1) == 0;
        SwizzledSmem current_stage = use_stage0 ? k_stage0 : k_stage1;
        SwizzledSmem next_stage = use_stage0 ? k_stage1 : k_stage0;
        flashinfer::cp_async::wait_group<0>();
        __syncthreads();
        if (tile + 1 < tile_count) {
            stage_pooled_keys_async(
                packed_k,
                next_stage,
                threadIdx.x,
                bh,
                (tile + 1) * kKeyBlocksPerTile,
                fine_blocks);
            flashinfer::cp_async::commit_group();
        }

        alignas(16) float score_frag[kFragmentElements];
        mma_qk_tile(current_stage, q_frag, score_frag, lane);

        // Each lane owns two Q centroids and two KV4 blocks. Their two key
        // representatives occupy fragment [4*key + 2*query + {0,1}].
#pragma unroll
        for (int query = 0; query < kQBlocksPerGroup; ++query) {
            const int64_t q_block = q_block_base + query;
            if (q_block >= q_blocks) continue;
            const int64_t row_base =
                ((bh * q_blocks + q_block) * kQueryCentroids + lane / 4) *
                fine_blocks;
#pragma unroll
            for (int key = 0; key < 2; ++key) {
                const int fragment = 4 * key + 2 * query;
                const int block = tile * kKeyBlocksPerTile + (lane & 3) + 4 * key;
                // Round before accumulating to match the stored LSE.
                const half stored = __float2half_rn(pair_lse(
                    score_frag[fragment] + kLogRepresentativeSize,
                    score_frag[fragment + 1] + kLogRepresentativeSize));
                store_block_lse(output, row_base, block, fine_blocks,
                                stored, maximum[query], denominator[query]);
            }
        }
        __syncwarp();
    }

    // The four lanes of a centroid hold disjoint key blocks; combine them.
#pragma unroll
    for (int mask = 1; mask <= 2; mask <<= 1) {
#pragma unroll
        for (int query = 0; query < kQBlocksPerGroup; ++query) {
            combine_lse(maximum[query], denominator[query],
                        __shfl_xor_sync(0xffffffffU, maximum[query], mask),
                        __shfl_xor_sync(0xffffffffU, denominator[query], mask));
        }
    }
    if ((lane & 3) == 0) {
        const int centroid = lane / 4;
#pragma unroll
        for (int query = 0; query < kQBlocksPerGroup; ++query) {
            if (q_block_base + query >= q_blocks) continue;
            const int64_t slot =
                (bh * q_blocks + q_block_base + query) * kQueryCentroids + centroid;
            row_max[slot] = maximum[query];
            row_den[slot] = denominator[query];
        }
    }
}


// ---- Population-weighted block scores -------------------------------------

// IBCS block scores S_ij: centroid attention averaged by cluster population.
__global__ __launch_bounds__(kScoreThreads)
void ibcs_scores_kernel(
    const half* __restrict__ lse,
    const half* __restrict__ populations,
    const float* __restrict__ lse_max,
    const float* __restrict__ lse_den,
    float* __restrict__ scores,
    int fine_blocks) {
    __shared__ float row_max[kQueryCentroids];
    __shared__ float row_den[kQueryCentroids];
    __shared__ float row_weight[kQueryCentroids];
    const int64_t row = blockIdx.x;
    if (threadIdx.x < kQueryCentroids) {
        const int centroid = threadIdx.x;
        row_max[centroid] = lse_max[row * kQueryCentroids + centroid];
        row_den[centroid] = lse_den[row * kQueryCentroids + centroid];
        row_weight[centroid] =
            __half2float(populations[row * kQueryCentroids + centroid]);
    }
    __syncthreads();

    float total_weight = 0.0f;
#pragma unroll
    for (int centroid = 0; centroid < kQueryCentroids; ++centroid) {
        total_weight += row_weight[centroid];
    }
    // The preceding kernel computed each centroid's softmax normalizer;
    // stream the stored block LSEs only once.
    for (int start = 0; start < fine_blocks; start += kScoreThreads) {
        const int fine_block = start + threadIdx.x;
        if (fine_block < fine_blocks) {
            float score = 0.0f;
#pragma unroll
            for (int centroid = 0; centroid < kQueryCentroids; ++centroid) {
                const float value = __half2float(
                    lse[(row * kQueryCentroids + centroid) * fine_blocks + fine_block]);
                const float probability =
                    exp2f(value - row_max[centroid]) / row_den[centroid];
                score += row_weight[centroid] * probability;
            }
            scores[row * fine_blocks + fine_block] = score / total_weight;
        }
    }
}

template <typename T>
T* tensor_ptr(const torch::Tensor& tensor) {
    return reinterpret_cast<T*>(tensor.data_ptr());
}

}  // namespace

// ---- Host orchestration ---------------------------------------------------

torch::Tensor ibcs_cuda(torch::Tensor q, torch::Tensor k) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q and k must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device(), "q and k must use the same CUDA device");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q and k must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes() && q.dim() == 4,
                "q and k must have matching [batch, heads, tokens, 128] shapes");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16 &&
                    k.scalar_type() == at::ScalarType::BFloat16,
                "IBCS requires bfloat16 q/k");
    TORCH_CHECK(q.size(3) == kHeadDim, "IBCS requires head_dim=128");
    TORCH_CHECK(q.size(2) % kSamplingBlockSize == 0,
                "sequence length must be divisible by 4");
    const int64_t batch = q.size(0);
    const int64_t heads = q.size(1);
    const int64_t batch_heads = batch * heads;
    const int64_t seq_len = q.size(2);
    const int64_t q_blocks = (seq_len + kQueryBlockSize - 1) / kQueryBlockSize;
    const int64_t fine_blocks = seq_len / kSamplingBlockSize;
    TORCH_CHECK(fine_blocks > 0 && fine_blocks <= 32768,
                "KV4 block count must be in [1, 32768]");
    const int64_t q_groups = (q_blocks + kQBlocksPerGroup - 1) / kQBlocksPerGroup;
    const int64_t rows = batch_heads * q_blocks;

    c10::cuda::CUDAGuard device_guard(q.device());
    // Phase 1/2: pack the irregular input into fixed-size GPU-friendly tiles.
    auto packed_q = torch::empty(
        {batch, heads, q_blocks, kQueryCentroids, kHeadDim}, q.options());
    auto q_populations = torch::empty(
        {batch, heads, q_blocks, kQueryCentroids}, q.options().dtype(torch::kFloat16));
    auto packed_k = torch::empty(
        {batch, heads, fine_blocks, kKeyRepresentatives, kHeadDim}, k.options());
    auto lse = torch::empty(
        {rows, kQueryCentroids, fine_blocks}, q.options().dtype(torch::kFloat16));
    auto lse_max = torch::empty(
        {rows, kQueryCentroids}, q.options().dtype(torch::kFloat32));
    auto lse_den = torch::empty_like(lse_max);
    auto scores = torch::empty(
        {rows, fine_blocks}, q.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 q_pack_grid(
        static_cast<unsigned int>(q_blocks), static_cast<unsigned int>(batch_heads));
    cluster_queries_kernel<<<q_pack_grid, kHeadDim, 0, stream>>>(
        tensor_ptr<const Bf16>(q), tensor_ptr<Bf16>(packed_q),
        tensor_ptr<half>(q_populations), seq_len, q_blocks);
    const dim3 k_pack_grid(
        static_cast<unsigned int>(fine_blocks), static_cast<unsigned int>(batch_heads));
    pool_keys_kernel<<<k_pack_grid, kHeadDim, 0, stream>>>(
        tensor_ptr<const Bf16>(k), tensor_ptr<Bf16>(packed_k), fine_blocks);

    // Phase 3: run the tiled centroid/key attention and retain its normalizers
    // for the final population-weighted reduction.
    const float qk_scale_log2 = static_cast<float>(
        std::pow(static_cast<double>(kHeadDim), -0.5) * 1.4426950408889634);
    const int q_group_tiles = static_cast<int>((q_groups + kWarps - 1) / kWarps);
    const dim3 score_grid(q_group_tiles, static_cast<unsigned int>(batch_heads));
    ibcs_lse_kernel<<<score_grid, kThreads, kSharedBytes, stream>>>(
        tensor_ptr<const Bf16>(packed_q), tensor_ptr<const Bf16>(packed_k),
        tensor_ptr<half>(lse),
        lse_max.data_ptr<float>(), lse_den.data_ptr<float>(),
        q_blocks, static_cast<int>(fine_blocks), qk_scale_log2);
    // Phase 4: turn the eight centroid distributions into one block mass.
    ibcs_scores_kernel<<<static_cast<unsigned int>(rows), kScoreThreads, 0, stream>>>(
        tensor_ptr<const half>(lse), tensor_ptr<const half>(q_populations),
        lse_max.data_ptr<float>(), lse_den.data_ptr<float>(),
        scores.data_ptr<float>(), static_cast<int>(fine_blocks));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}
