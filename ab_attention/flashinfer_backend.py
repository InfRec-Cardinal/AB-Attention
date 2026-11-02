"""Hopper execution through FlashInfer's paged sparse prefill backend.

AB-Attention owns the page table and SIP weights.  FlashInfer owns the
Hopper-specific gather, pipeline, WGMMA, and online-softmax implementation.
The only CUDA code kept here is the small semantic adapter that adds one SIP
log2 weight for each selected KV4 block.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import weakref

import torch


_HEAD_DIM = 128
_QUERY_BLOCK = 128
_PAGE_SIZE = 4
_WORKSPACE_BYTES = 128 * 1024 * 1024


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Inference tensors have no version counter; their plans are immutable."""
    return None if tensor.is_inference() else tensor._version


_SIP_VARIANT_DECL = r"""
#include <flashinfer/attention/hopper/variants.cuh>

namespace flashinfer {

struct ABSipAttention {
  const float* sip_log_weights;
  float sm_scale_log2;

  template <typename MainloopParams, typename BlockCoord>
  __device__ ABSipAttention(const MainloopParams& params, const BlockCoord& block_coord) {
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    sip_log_weights = params.additional_params.maybe_sip_log_weights +
                      batch_idx * params.additional_params.sip_log_stride;
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(sm_scale_log2);
  }

  REGISTER_LOGITS_TRANSFORM(
      params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
      { return logits + sip_log_weights[kv_idx / 4] / sm_scale_log2; })
};

}  // namespace flashinfer
"""


def _global_page_indices(
    descriptor: torch.Tensor,
    *,
    batch_heads: int,
    query_blocks: int,
    fine_blocks: int,
) -> torch.Tensor:
    """Translate per-head KV4 ids into the shared paged-cache address space."""

    local_ids = descriptor.reshape(batch_heads, query_blocks, -1).to(torch.int32)
    head_offsets = torch.arange(
        batch_heads, device=descriptor.device, dtype=torch.int32) * fine_blocks
    return (local_ids + head_offsets[:, None, None]).flatten()


def _pack_queries(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack head-major query blocks, padding only the final block if needed."""

    batch, heads, tokens, dim = q.shape
    batch_heads = batch * heads
    query_blocks = math.ceil(tokens / _QUERY_BLOCK)
    padded_tokens = query_blocks * _QUERY_BLOCK
    flat = q.reshape(batch_heads, tokens, dim)
    if padded_tokens != tokens:
        padded = torch.zeros(
            (batch_heads, padded_tokens, dim), dtype=q.dtype, device=q.device
        )
        padded[:, :tokens].copy_(flat)
        flat = padded
    return flat.reshape(batch_heads * query_blocks * _QUERY_BLOCK, 1, dim), query_blocks


@dataclass
class _Runner:
    wrapper: object
    page_indices: torch.Tensor
    page_index_source: weakref.ReferenceType | None = None
    page_index_version: int | None = None
    page_index_stream: int | None = None


@lru_cache(maxsize=2)
def _get_runner(
    device_index: int,
    batch_heads: int,
    query_blocks: int,
    samples: int,
    dtype: torch.dtype,
) -> _Runner:
    """Create one reusable FlashInfer plan for a fixed tensor shape."""

    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    device = torch.device("cuda", device_index)
    rows = batch_heads * query_blocks
    page_indices = torch.zeros(rows * samples, dtype=torch.int32, device=device)
    row_ids = torch.arange(rows + 1, dtype=torch.int32, device=device)
    qo_indptr = row_ids * _QUERY_BLOCK
    paged_kv_indptr = row_ids * samples
    last_page_len = torch.full((rows,), _PAGE_SIZE, dtype=torch.int32, device=device)
    workspace = torch.empty(_WORKSPACE_BYTES // 4, dtype=torch.float32, device=device)
    # Keep the SIP row stride runtime-valued so every sampling budget reuses
    # one FlashInfer variant instead of creating a new template specialization.
    jit_args = [
        f"ab_attention_sip_d{_HEAD_DIM}",
        dtype, dtype, dtype, torch.int32, _HEAD_DIM, _HEAD_DIM,
        ["maybe_sip_log_weights"], ["float"],
        ["sm_scale", "sip_log_stride"], ["double", "int"],
        "ABSipAttention", _SIP_VARIANT_DECL,
    ]
    with torch.cuda.device(device):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=qo_indptr,
            paged_kv_indptr_buf=paged_kv_indptr,
            paged_kv_indices_buf=page_indices,
            paged_kv_last_page_len_buf=last_page_len,
            backend="fa3",
            jit_args=jit_args,
        )
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            page_indices,
            last_page_len,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=_HEAD_DIM,
            head_dim_vo=_HEAD_DIM,
            page_size=_PAGE_SIZE,
            causal=False,
            sm_scale=1.0 / math.sqrt(_HEAD_DIM),
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            disable_split_kv=True,
        )
    return _Runner(wrapper, page_indices)


@torch.no_grad()
def paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    descriptor: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    return_lse: bool = False,
):
    """Run weighted sparse attention while keeping Hopper details in FlashInfer."""

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V must have the same [B,H,N,D] shape")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if q.device.type != "cuda":
        raise ValueError("FlashInfer Hopper attention requires CUDA tensors")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("the FlashInfer Hopper backend currently requires BF16 Q/K/V")
    if q.shape[-1] != _HEAD_DIM or q.shape[2] % _PAGE_SIZE:
        raise ValueError("FlashInfer Hopper attention requires head_dim=128 and N divisible by 4")
    batch, heads, tokens, dim = q.shape
    batch_heads = batch * heads
    rows = batch_heads * math.ceil(tokens / _QUERY_BLOCK)
    if (descriptor.ndim != 3 or descriptor.shape != log_weights.shape
            or descriptor.shape[0] != rows or descriptor.shape[-1] != 32):
        raise ValueError(f"descriptor and log_weights must share [{rows}, pages, 32] shape")
    if descriptor.device != q.device or log_weights.device != q.device:
        raise ValueError("descriptor and log_weights must be on the Q device")

    query, query_blocks = _pack_queries(q.contiguous())
    fine_blocks = tokens // _PAGE_SIZE
    samples = descriptor.shape[1] * 32
    runner = _get_runner(
        q.device.index, batch_heads, query_blocks, samples, q.dtype,
    )
    # Plans persist across diffusion steps, so translate their page table once.
    current_stream = int(torch.cuda.current_stream(q.device).cuda_stream)
    source_version = _tensor_version(descriptor)
    if not (
        runner.page_index_source is not None
        and runner.page_index_source() is descriptor
        and runner.page_index_version == source_version
        and runner.page_index_stream == current_stream
    ):
        global_ids = _global_page_indices(
            descriptor,
            batch_heads=batch_heads,
            query_blocks=query_blocks,
            fine_blocks=fine_blocks,
        )
        runner.page_indices.copy_(global_ids, non_blocking=True)
        runner.page_index_source = weakref.ref(descriptor)
        runner.page_index_version = source_version
        runner.page_index_stream = current_stream

    k_pages = k.contiguous().view(-1, _PAGE_SIZE, 1, dim)
    v_pages = v.contiguous().view(-1, _PAGE_SIZE, 1, dim)
    weights = log_weights.reshape(-1).to(torch.float32).contiguous()
    sm_scale = 1.0 / math.sqrt(dim)
    result = runner.wrapper.run(
        query,
        (k_pages, v_pages),
        weights,
        sm_scale,
        samples,
        return_lse=return_lse,
    )
    if return_lse:
        output, lse = result
    else:
        output = result

    output = output.reshape(batch_heads, query_blocks * _QUERY_BLOCK, dim)[
        :, :tokens
    ].reshape_as(q)
    if not return_lse:
        return output
    lse = lse.reshape(batch_heads, query_blocks * _QUERY_BLOCK)[:, :tokens]
    return output, lse.reshape(batch, heads, tokens)
