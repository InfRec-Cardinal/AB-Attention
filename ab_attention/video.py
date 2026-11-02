"""Video integration: SVG2 preprocessing, AB video attention, exact text attention.

The paper schedule keeps the first 10 steps and layer 0 dense. Other layers
refresh their plan on steps 11, 15, ..., 47 and reuse it between refreshes.
"""
from types import MethodType

import torch

from .attention import SamplePlan, apply_plan, build_plan


def _exact(q, k, v, *, return_lse=False):
    """FA3 attention on [B,H,N,D]; FlashInfer returns base-2 LSE."""
    import flashinfer

    batch, heads, tokens, dim = q.shape
    output = flashinfer.single_prefill_with_kv_cache(
        q.flatten(0, 1).transpose(0, 1).contiguous(),
        k.flatten(0, 1).transpose(0, 1).contiguous(),
        v.flatten(0, 1).transpose(0, 1).contiguous(),
        causal=False, backend="fa3", return_lse=return_lse,
    )
    if return_lse:
        output, lse = output
        return (output.transpose(0, 1).reshape(batch, heads, tokens, dim),
                lse.transpose(0, 1).reshape(batch, heads, tokens))
    return output.transpose(0, 1).reshape(batch, heads, tokens, dim)


def _merge(video_state, text_state):
    """Merge disjoint key partitions with FlashInfer's CUDA state merger."""
    import flashinfer

    args = []
    for output, lse in (video_state, text_state):
        args.extend((output[0].transpose(0, 1).contiguous(),
                     lse[0].transpose(0, 1).contiguous()))
    output, _ = flashinfer.merge_state(*args)
    return output.transpose(0, 1).unsqueeze(0)


def _store_plan(plan, previous):
    """Keep per-layer plans in pinned RAM so the video model fits on one GPU."""
    if previous is None:
        previous = SamplePlan(*(torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                                for t in plan))
    for storage, tensor in zip(previous, plan):
        storage.copy_(tensor)
    return previous


class VideoAttention:
    """One pipeline's denoising schedule and per-layer sample plans.

    Call ``reset()`` before each video and pass ``step_end`` as Diffusers'
    ``callback_on_step_end``. Wan's two sequential CFG calls share one plan.
    """

    def __init__(self, sample_pages, *, dense=False):
        self.sample_pages = sample_pages
        self.dense = dense
        self.reset()

    def reset(self):
        self.step = 1
        self.plans = {}

    def step_end(self, pipe, index, timestep, callback_kwargs):
        self.step = index + 2  # Diffusers indices are zero-based.
        return callback_kwargs

    def is_sparse(self, layer):
        return not self.dense and self.step > 10 and layer > 0

    def attend(self, layer, q, k, v, *, return_lse=False):
        anchor = 11 + (self.step - 11) // 4 * 4
        stored_anchor, stored = self.plans.get(layer, (None, None))
        if stored_anchor != anchor:
            seed = 42 + layer * 1_000_003 + anchor * 97
            plan = build_plan(q, k, sample_pages=self.sample_pages, seed=seed)
        else:
            plan = SamplePlan(*(t.to(q.device, non_blocking=True) for t in stored))
        output = apply_plan(q, k, v, plan, return_lse=return_lse)
        if stored_anchor != anchor:
            self.plans[layer] = (anchor, _store_plan(plan, stored))
        return output


def use_cube_layout(pipe, schedule, *, height, width, num_frames):
    """Group video tokens into 4×4×4 cubes for AB's sparse steps.

    Reorder once before the first block, carry the original RoPE positions,
    and restore raster order before unpatchifying. Text tokens stay in place.
    Install once, with the same video shape used for generation.
    """
    shape = (1 + (num_frames - 1) // 4, height // 16, width // 16)
    t, h, w = shape
    grid = torch.arange(t * h * w).reshape(shape)
    # Pad only the index grid; boundary cubes contain their valid tokens only.
    grid = torch.nn.functional.pad(grid, (0, -w % 4, 0, -h % 4, 0, -t % 4), value=-1)
    nt, nh, nw = (size // 4 for size in grid.shape)
    order = grid.reshape(nt, 4, nh, 4, nw, 4).permute(0, 2, 4, 1, 3, 5).reshape(-1)
    order = order[order >= 0].to(pipe.device)
    inverse = order.argsort()
    transformer = pipe.transformer
    hunyuan = hasattr(transformer, "transformer_blocks")
    first = transformer.transformer_blocks[0] if hunyuan else transformer.blocks[0]

    def enter(module, args):
        if schedule.is_sparse(1):
            return (args[0].index_select(1, order), *args[1:])

    def rope(module, args, output):
        if schedule.is_sparse(1):
            if hunyuan:
                return tuple(x.index_select(0, order) for x in output)
            return output.index_select(-2, order)

    def leave(module, args, output):
        if schedule.is_sparse(1):
            return output.index_select(1, inverse)

    return (first.register_forward_pre_hook(enter),
            transformer.rope.register_forward_hook(rope),
            transformer.proj_out.register_forward_hook(leave))


def _hunyuan_attention(schedule, layer, q, k, v, video_tokens, prompt_tokens):
    """Video↔text is exact; only video→video uses the sampled plan."""
    prompt_end = video_tokens + prompt_tokens
    if schedule.is_sparse(layer):
        video_state = schedule.attend(
            layer, q[:, :, :video_tokens].contiguous(),
            k[:, :, :video_tokens].contiguous(),
            v[:, :, :video_tokens].contiguous(), return_lse=True,
        )
        text_state = _exact(q[:, :, :video_tokens],
                            k[:, :, video_tokens:prompt_end],
                            v[:, :, video_tokens:prompt_end], return_lse=True)
        outputs = [_merge(video_state, text_state),
                   _exact(q[:, :, video_tokens:prompt_end],
                          k[:, :, :prompt_end], v[:, :, :prompt_end])]
    else:
        outputs = [_exact(q[:, :, :prompt_end], k[:, :, :prompt_end],
                          v[:, :, :prompt_end])]
    # Padding tokens form their own attention partition in HunyuanVideo.
    if prompt_end < q.shape[2]:
        outputs.append(_exact(q[:, :, prompt_end:], k[:, :, prompt_end:],
                              v[:, :, prompt_end:]))
    return torch.cat(outputs, dim=2)


def install_wan(pipe, *, sample_pages=64, dense=False):
    """Install AB in a Diffusers Wan2.1 T2V pipeline; return its schedule."""
    from svg.models.wan import attention, custom_models

    if not attention.ENABLE_FAST_KERNEL:
        raise RuntimeError("Build SVG2's CUDA kernels first; see README.md#installation.")
    schedule = VideoAttention(sample_pages, dense=dense)

    class WanProcessor(attention.WanAttn_SVGAttn_Processor2_0):
        def attention_core_logic(self, query, key, value, timestep):
            if schedule.is_sparse(self.layer_idx):
                return schedule.attend(self.layer_idx, query, key, value)
            return _exact(query, key, value)

    transformer = pipe.transformer
    transformer.forward = MethodType(
        custom_models.WanTransformer3DModel_Sparse.forward, transformer)
    for index, block in enumerate(transformer.blocks):
        block.forward = MethodType(custom_models.WanTransformerBlock_Sparse.forward, block)
        block.attn1.set_processor(WanProcessor(index))
    return schedule


def install_hunyuan(pipe, *, height=720, width=1280, num_frames=129,
                    sample_pages=71, dense=False):
    """Install AB in Diffusers HunyuanVideo 1.0 (one video per call).

    Call ``schedule.set_prompt(prompt)`` before generating each new prompt.
    Use ``max_sequence_length=256`` and ``true_cfg_scale=1`` as in the paper.
    """
    from svg.models.hyvideo import attention, custom_models
    from svg.models.hyvideo.utils import get_prompt_length

    if not attention.ENABLE_FAST_KERNEL:
        raise RuntimeError("Build SVG2's CUDA kernels first; see README.md#installation.")

    class HunyuanSchedule(VideoAttention):
        def set_prompt(self, prompt):
            self.reset()
            self.prompt_tokens = int(get_prompt_length(pipe, prompt, device="cpu"))

    schedule = HunyuanSchedule(sample_pages, dense=dense)
    video_tokens = (1 + (num_frames - 1) // 4) * (height // 16) * (width // 16)

    class HunyuanProcessor(attention.Hunyuan_SVGAttn_Processor2_0):
        def attention_core_logic(self, query, key, value, timestep, layer_idx, cu_max_seqlens):
            if query.shape[0] != 1 or query.shape[2] != video_tokens + 256:
                raise ValueError("Use one video with the installed resolution and 256 text tokens.")
            return _hunyuan_attention(schedule, layer_idx, query, key, value,
                                      video_tokens, schedule.prompt_tokens)

    transformer = pipe.transformer
    transformer.forward = MethodType(
        custom_models.HunyuanVideoTransformer3DModel_Sparse.forward, transformer)
    index = 0
    for blocks, forward in (
        (transformer.transformer_blocks,
         custom_models.HunyuanVideoTransformerBlock_Sparse.forward),
        (transformer.single_transformer_blocks,
         custom_models.HunyuanVideoSingleTransformerBlock_Sparse.forward),
    ):
        for block in blocks:
            block.forward = MethodType(forward, block)
            block.attn.set_processor(HunyuanProcessor(index))
            index += 1
    return schedule
