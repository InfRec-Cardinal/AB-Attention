<p align="center">
  <a href="https://www.cardinal-agi.com/">
    <img src="https://infrec-cardinal.github.io/AB-Attention/static/images/ab-attention-logo.svg" alt="AB-Attention by Cardinal AI" width="560">
  </a>
</p>

<p align="center">
  <strong>Official repository for AB-Attention</strong>
</p>

<p align="center">
  Accurate Block Attention for Accelerating Video Diffusion Transformers
</p>

<p align="center">
  <a href="https://infrec-cardinal.github.io/AB-Attention/static/papers/ab-attention.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square&amp;logo=adobeacrobatreader&amp;logoColor=white" alt="Paper PDF"></a>
  <a href="https://arxiv.org/abs/2610.39663"><img src="https://img.shields.io/badge/arXiv-2610.39663-b31b1b?style=flat-square&amp;logo=arxiv&amp;logoColor=white" alt="arXiv paper"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Code-Available-2ea44f?style=flat-square&amp;logo=github&amp;logoColor=white" alt="Code available"></a>
  <a href="https://infrec-cardinal.github.io/AB-Attention/"><img src="https://img.shields.io/badge/Project_Page-2588a6?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="Project page"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-6f42c1?style=flat-square&amp;logo=readthedocs&amp;logoColor=white" alt="Citation"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> &nbsp;/&nbsp;
  <a href="#video-level-results">Results</a> &nbsp;/&nbsp;
  <a href="#how-it-works">Method</a> &nbsp;/&nbsp;
  <a href="#citation">Citation</a>
</p>

---

AB-Attention is a training-free sparse attention method for long video sequences. It combines local query clustering, inclusion-probability weighting, and paged GPU execution so that high sparsity does not require dropping the contribution of every unselected block from the attention estimate.

The implementation supports **Wan2.1 T2V 1.3B / 14B** and **HunyuanVideo 1.0 13B**
on NVIDIA Hopper GPUs, including H20 and H100.

<p align="center">
  <strong>89–92% attention skipped</strong> &nbsp;·&nbsp; <strong>up to 2.23× end-to-end speedup</strong> &nbsp;·&nbsp; <strong>matched video evaluation</strong>
</p>

## Quick start

```bash
git clone --single-branch https://github.com/InfRec-Cardinal/AB-Attention.git
cd AB-Attention
```

<details>
<summary><strong>Install dependencies and kernels</strong> (Linux · CUDA 12.4 · Hopper GPU)</summary>

### Installation

Use Linux, Python 3.10+, PyTorch 2.6 with CUDA 12.4, the CUDA 12.4 toolkit,
GCC/G++ 11 or 12, and enough GPU memory for the model (the paper uses 96 GB H20).
Activate a fresh Python environment and set `CUDA_HOME` to the toolkit directory.
From the AB-Attention checkout, install the video dependencies and kernels:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ninja 'setuptools>=77' wheel
pip install --no-build-isolation -e '.[video]' --extra-index-url https://pypi.nvidia.com

# Keep the upstream dependency outside this checkout.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/svg-project/Sparse-VideoGen.git ../Sparse-VideoGen
GIT_LFS_SKIP_SMUDGE=1 git -C ../Sparse-VideoGen checkout f89aedaf169ac2ae5b186bda674e53c3dc08c476
git -C ../Sparse-VideoGen submodule update --init \
  svg/kernels/3rdparty/cutlass svg/kernels/3rdparty/flashinfer
pip install -e ../Sparse-VideoGen
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Build SVG2's unchanged Norm/RoPE source with PyTorch's extension tooling.
python - <<'PY'
from pathlib import Path
from torch.utils.cpp_extension import load

root = Path('../Sparse-VideoGen/svg/kernels').resolve()
build = root / 'build'
build.mkdir(exist_ok=True)
load('_kernels', sources=[str(root / 'csrc/ops.cu')],
     extra_include_paths=[str(root / p) for p in
                          ('include', '3rdparty/cutlass/include',
                           '3rdparty/flashinfer/include')],
     extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a',
                       '-U__CUDA_NO_HALF_CONVERSIONS__',
                       '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                       '--expt-extended-lambda', '--expt-relaxed-constexpr',
                       '--use_fast_math'],
     build_directory=str(build))
PY
export PYTHONPATH="$(realpath ../Sparse-VideoGen/svg/kernels/build):${PYTHONPATH:-}"
```

SVG2 imports FlashAttention and cuVS even though AB does not run its KMeans
selector. AB installs standard FlashInfer 0.6.12 as a runtime dependency; no SVG2
FlashInfer fork or sparse-attention patch is needed.

For the Q/K/V API alone, install `.` instead of `.[video]` after installing
PyTorch and the build tools; SVG2 and the video dependencies are not needed.

</details>

After installation, generate a video:

```bash
python generate.py --model wan1.3b \
  --prompt 'A penguin walks across a snowy beach.' --output outputs/wan.mp4
```

Use `--model wan14b` or `--model hunyuan` for the other models, `--dense`
for the full-attention baseline, or `--cube-layout` for optional 4×4×4
spatiotemporal grouping.

<details>
<summary><strong>Paper settings, timing, and pipeline integration</strong></summary>

Models download from Hugging Face. `--model-path /path/to/model` accepts an
existing Diffusers-format checkpoint; for Hunyuan this must include its
transformer, VAE, both text encoders, and tokenizers. The default Hunyuan
download is the community Diffusers conversion. For paper reproduction, use
the same checkpoint conversion/revision, prompt and initial noise as the
reference, rather than treating any downloadable checkpoint as equivalent.

| Setting | Wan2.1 | HunyuanVideo |
| --- | --- | --- |
| Resolution | 1280 × 720 | 1280 × 720 |
| Frames / denoising steps | 81 / 50 | 129 / 50 |
| Scheduler | UniPC, flow shift 5 | FlowMatch Euler, shift 7 |
| Guidance | CFG 5 | Embedded guidance 6, true CFG 1 |
| Sample pages / keys | 64 / 8192 | 71 / 9088 |
| Text sequence length | 512 | 256 |

Both use BF16 inference, Dense FA3 for steps 1–10 and layer 0 at every step,
and plan refreshes at steps 11, 15, …, 47. Wan shares the conditional branch's
plan with the unconditional branch. Hunyuan keeps video/text interactions
exact and preserves the separate padding partition. Plans live in pinned CPU
memory between attention calls.

`--dense` uses full FA3 attention with the same model preprocessing.
`--warmup` runs a complete video before reporting text encoding + denoising +
VAE time; model loading and MP4 export are excluded. Compare the same model,
prompt, negative prompt, resolution, frames, steps and initial noise on the
same idle GPU. `--latents noise.npy` accepts an explicit float32 initial latent;
otherwise the script generates CPU noise with `--seed` (default 42).
The default negative prompt is empty; pass the reference's negative prompt
when reproducing an existing video.

For a quick functional check, use
`--height 256 --width 256 --frames 17 --steps 15 --sample-pages 1`.
This exercises Dense, plan refresh and reuse;
it is not a paper quality or speed measurement.

### Optional cube layout

Add `--cube-layout` to group video tokens into 4×4×4 spatiotemporal cubes.
The first ten denoising steps are unchanged. From step 11, tokens and their
RoPE positions are reordered once at transformer entry, then restored at exit.
The IBCS/SIP kernels and sample budget are unchanged. This optional extension
follows the permutation described in [VSA2](https://arxiv.org/abs/2609.32882)
and the boundary-token ordering used in [FastVideo](https://github.com/hao-ai-lab/FastVideo);
the paper's reported results use the original raster order.

For an existing pipeline, call `use_cube_layout` from `ab_attention.video`
after installing AB, passing the same `height`, `width`, and `num_frames` as
the generation call.

### Use an existing pipeline

```python
from ab_attention.video import install_wan

schedule = install_wan(pipe)
schedule.reset()  # Start each new video with fresh plans.
video = pipe(prompt=prompt, num_inference_steps=50,
             callback_on_step_end=schedule.step_end).frames[0]
```

For Hunyuan, call `install_hunyuan(pipe, height=720, width=1280, num_frames=129)`,
then `schedule.set_prompt(prompt)` before each call. Pass that same shape,
`max_sequence_length=256`, `true_cfg_scale=1`, and the callback to `pipe`.
Use the scheduler and VAE settings in `generate.py` to match the table.
Installers change only the supplied pipeline's forwards and processors.

</details>

The attention API also works directly with BF16 tensors shaped `[B, H, N, 128]`:

```python
from ab_attention import build_plan, apply_plan

plan = build_plan(q, k, sample_pages=64)
output = apply_plan(q, k, v, plan)
```

IBCS and SIP use CUDA kernels; the paged attention backend reuses FlashInfer.

## Video-level results

The results below are the paper's end-to-end video measurements on the 600-prompt Penguin Benchmark. Quality is compared with matched dense references; speed is measured against matched dense FlashAttention-3 runs on NVIDIA H20.

| Model | Attention skipped | PSNR | SSIM | LPIPS | End-to-end speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wan2.1-T2V · 1.3B | 89.16% | 34.436 | 0.9328 | 0.0734 | **2.01x** |
| Wan2.1-T2V · 14B | 89.16% | 30.788 | 0.9169 | 0.0964 | **1.84x** |
| HunyuanVideo · 13B | 92.35% | 34.347 | 0.9480 | 0.0700 | **2.23x** |

### Same prompt, dense vs. accelerated

The two videos below use the same prompt and generation settings. Play them
directly in the README; the project page contains the full multi-method gallery.

<table>
  <tr>
    <th align="center">Dense attention · original</th>
    <th align="center">AB-Attention · accelerated</th>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/ceb7c049-2812-43d0-868b-c739f5ff982b" poster="https://infrec-cardinal.github.io/AB-Attention/static/images/magpie-dense.webp" controls muted loop playsinline preload="metadata" width="100%" aria-label="Magpie generated with dense attention"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/775f227d-857f-4425-9b79-f9c6435419e6" poster="https://infrec-cardinal.github.io/AB-Attention/static/images/magpie-ab-attention.webp" controls muted loop playsinline preload="metadata" width="100%" aria-label="Magpie generated with AB-Attention"></video>
    </td>
  </tr>
</table>

See the [project page](https://infrec-cardinal.github.io/AB-Attention/) for matched dense, AB-Attention, DFSAttn, and SVG2 videos.

## How it works

<p align="center">
  <img src="https://infrec-cardinal.github.io/AB-Attention/static/images/overview.png" alt="AB-Attention overview: local block scoring, probability-corrected sampling, and tiled GPU execution" width="100%">
</p>

1. **Intra-Block Clustering Selector (IBCS)** scores small key/value blocks from query representatives formed within each query block.
2. **Sampling with Inclusion-Probability Weighting (SIP)** samples blocks without replacement and reweights their contribution to the attention normalization.
3. **Paged execution** gathers selected blocks into full attention tiles for efficient GPU execution.

### Reading the implementation

1. [`generate.py`](generate.py): load a model → install AB → generate/export a video.
2. [`ab_attention/video.py`](ab_attention/video.py): Dense schedule, plan reuse, model integration.
3. [`ab_attention/attention.py`](ab_attention/attention.py): `build_plan` → `apply_plan`.
4. [`csrc/ibcs.cu`](csrc/ibcs.cu) and [`csrc/sip.cu`](csrc/sip.cu): scoring and sampling.
5. [`ab_attention/flashinfer_backend.py`](ab_attention/flashinfer_backend.py):
   convert the sampled blocks to FlashInfer pages and apply SIP logit weights.

## Citation

```bibtex
@misc{xu2026abattentionaccurateblockattention,
  title = {AB-Attention: Accurate Block Attention for Accelerating Video Diffusion Transformers},
  author = {Bin Xu and Mo Zhang and Zhengfeiyang Zhang and Xinfeng Yang and
            Yixing Ma and Wei Wang and HeXin Wang and Pengfei Zhou and Wangbo Zhao},
  year = {2026},
  eprint = {2610.39663},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2610.39663}
}
```

## License and acknowledgments

Released under the [MIT License](LICENSE). Inspired by
[DFSAttn](https://github.com/jessica-hujie/DFSAttn) and
[Sparse-VideoGen (SVG2)](https://github.com/svg-project/Sparse-VideoGen).
The three vendored [FlashInfer headers](csrc/flashinfer/) come from
[revision f6717ff](https://github.com/flashinfer-ai/flashinfer/tree/f6717ff6bc6061c4eb0474576746ee1b42bd6325/include/flashinfer)
and retain their [Apache-2.0 license](csrc/flashinfer/LICENSE).
