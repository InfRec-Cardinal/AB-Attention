"""AB-Attention and explicit sampling-plan reuse."""
import math
from typing import NamedTuple

import torch

# Bound IBCS temporary storage while keeping several heads per launch.
_LSE_BUDGET_BYTES = 1536 * 1024 * 1024


class SamplePlan(NamedTuple):
    """SIP sample and log2(1/pi) weights, shaped [rows, pages, 32].

    ``descriptor``  — int32, sampled KV4 block ids, shape [rows, pages, 32].
    ``log_weights`` — float32, log2 inverse-inclusion weights, same shape.

    ``rows`` is batch * heads * ceil_div(tokens, 128); ``pages`` is
    ``sample_pages``; each page packs 32 sampled KV4 blocks.
    Treat both tensors as read-only; create a new plan when refreshing selection.
    """

    descriptor: torch.Tensor
    log_weights: torch.Tensor


@torch.no_grad()
def build_plan(q, k, *, sample_pages=64, seed=42):
    """Score and sample ``sample_pages * 128`` keys per 128-query block.

    Q and K must have shape [batch, heads, tokens, 128], BF16 dtype, and live
    on the same Hopper GPU. Tokens must be divisible by 4 and exceed the
    sampled key count.

    IBCS scores KV4 blocks via Q8 cluster centroids and pooled key pairs;
    SIP draws ``sample_pages * 32`` blocks without replacement and returns
    log2(1/pi) weights used as logit shifts at attention time.

    Returns a ``SamplePlan`` with both tensors shaped [rows, pages, 32].
    """
    sample_pages = int(sample_pages)
    if q.ndim != 4 or q.shape != k.shape or q.shape[2] % 4:
        raise ValueError("Q/K must match [B,H,N,D] with N divisible by 4")
    if sample_pages <= 0:
        raise ValueError("sample_pages must be positive")
    if sample_pages * 128 >= q.shape[2]:
        raise ValueError("sampled key count must be smaller than the sequence length")
    from . import _C as ops

    batch_heads = q.shape[0] * q.shape[1]
    tokens, dim = q.shape[2:]
    q_blocks = math.ceil(tokens / 128)
    rows, samples = batch_heads * q_blocks, sample_pages * 32

    # Draw all offsets before chunking so the per-head launch count does not
    # affect the sampling sequence.
    generator = torch.Generator(device=q.device).manual_seed(seed)
    offsets = torch.rand(rows, device=q.device, generator=generator)
    flat_q = q.reshape(batch_heads, tokens, dim)
    flat_k = k.reshape(batch_heads, tokens, dim)

    # IBCS allocates [rows, Q_centroids, KV4_blocks] in fp16 as scratch.
    # Q_centroids=8, fp16=2 bytes, KV4_blocks = tokens//4.
    lse_bytes_per_head = q_blocks * 8 * (tokens // 4) * 2
    heads_per_chunk = max(1, min(batch_heads, _LSE_BUDGET_BYTES // lse_bytes_per_head))
    sampled = torch.empty((rows, samples), dtype=torch.int32, device=q.device)
    log_weights = torch.empty((rows, samples), dtype=torch.float32, device=q.device)

    for start in range(0, batch_heads, heads_per_chunk):
        stop = min(batch_heads, start + heads_per_chunk)
        first, last = start * q_blocks, stop * q_blocks
        scores = ops.ibcs(
            flat_q[start:stop].unsqueeze(0).contiguous(),
            flat_k[start:stop].unsqueeze(0).contiguous(),
        )
        ids, logs = ops.sip(scores, offsets[first:last], samples)
        sampled[first:last].copy_(ids)
        log_weights[first:last].copy_(logs)

    # Both fields share the same [rows, pages, 32] layout so apply_plan can
    # reshape them identically without a .contiguous() copy.
    return SamplePlan(
        sampled.reshape(rows, sample_pages, 32),
        log_weights.reshape(rows, sample_pages, 32),
    )


@torch.no_grad()
def apply_plan(q, k, v, plan, *, return_lse=False):
    """Run fine-grained selection through the Hopper paged-attention backend.

    A plan can be reused across denoising steps as long as the tensor shape
    and device stay the same.  The optional log2 softmax normalizer (LSE) is
    returned as a [batch, heads, tokens] float32 tensor, which callers can use
    to merge attention partitions.
    """
    from .flashinfer_backend import paged_attention

    return paged_attention(q, k, v, *plan, return_lse=return_lse)


@torch.no_grad()
def ab_attention(q, k, v, *, sample_pages=64, seed=42):
    """AB-Attention with a fresh sampling plan; output has Q's shape and dtype."""
    return apply_plan(q, k, v, build_plan(q, k, sample_pages=sample_pages, seed=seed))
