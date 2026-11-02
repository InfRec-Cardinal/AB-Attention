"""Generate a Wan2.1 or HunyuanVideo video with AB-Attention or matched FA3."""
import argparse
from pathlib import Path
import time

import torch

from ab_attention.video import install_hunyuan, install_wan, use_cube_layout


MODELS = {
    "wan1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "hunyuan": "hunyuanvideo-community/HunyuanVideo",
}


def load_pipeline(model, model_path):
    """Load the standard Diffusers components with the paper's scheduler."""
    if model == "hunyuan":
        from diffusers import HunyuanVideoPipeline, FlowMatchEulerDiscreteScheduler

        pipe = HunyuanVideoPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0))
        pipe.vae.enable_tiling()
    else:
        from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler

        vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True,
            num_train_timesteps=1000, flow_shift=5.0)
        pipe.vae.enable_tiling(tile_sample_min_height=192, tile_sample_min_width=192,
                               tile_sample_stride_height=128, tile_sample_stride_width=128)
    return pipe.to("cuda")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-path", help="Diffusers model directory or Hugging Face repo")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", type=Path, default=Path("outputs/video.mp4"))
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, help="Default: Wan 81; Hunyuan 129")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-pages", type=int, help="Default: Wan 64; Hunyuan 71")
    parser.add_argument("--latents", type=Path, help="Optional initial noise as a float32 .npy tensor")
    parser.add_argument("--dense", action="store_true", help="Matched full-attention FA3 baseline")
    parser.add_argument("--cube-layout", action="store_true", help="Group video tokens into 4x4x4 cubes")
    parser.add_argument("--warmup", action="store_true", help="Run a complete untimed video first")
    args = parser.parse_args()
    hunyuan = args.model == "hunyuan"
    frames = args.frames if args.frames is not None else (129 if hunyuan else 81)
    pages = args.sample_pages if args.sample_pages is not None else (71 if hunyuan else 64)
    if args.height % 16 or args.width % 16 or frames % 4 != 1:
        parser.error("height/width must be multiples of 16; frames must be 4n+1")
    tokens = (1 + (frames - 1) // 4) * (args.height // 16) * (args.width // 16)
    if not args.dense and (tokens % 4 or not 0 < pages * 128 < tokens):
        parser.error("AB needs video tokens divisible by 4 and 0 < sample_pages*128 < video tokens")

    pipe = load_pipeline(args.model, args.model_path or MODELS[args.model])
    if hunyuan:
        schedule = install_hunyuan(pipe, height=args.height, width=args.width,
                                    num_frames=frames, sample_pages=pages, dense=args.dense)
    else:
        schedule = install_wan(pipe, sample_pages=pages, dense=args.dense)
    if args.cube_layout:
        use_cube_layout(pipe, schedule, height=args.height, width=args.width, num_frames=frames)
    latent_shape = (1, pipe.transformer.config.in_channels,
                    1 + (frames - 1) // pipe.vae_scale_factor_temporal,
                    args.height // pipe.vae_scale_factor_spatial,
                    args.width // pipe.vae_scale_factor_spatial)
    if args.latents:
        import numpy as np

        latents = torch.from_numpy(np.load(args.latents, allow_pickle=False))
        if tuple(latents.shape) != latent_shape or latents.dtype != torch.float32:
            parser.error(f"initial noise must be float32 with shape {latent_shape}")
    else:
        latents = torch.randn(latent_shape, generator=torch.Generator().manual_seed(args.seed))
    options = dict(prompt=args.prompt, negative_prompt=args.negative_prompt,
                   height=args.height, width=args.width, num_frames=frames,
                   num_inference_steps=args.steps, guidance_scale=6.0 if hunyuan else 5.0,
                   max_sequence_length=256 if hunyuan else 512,
                   callback_on_step_end=schedule.step_end)
    if hunyuan:
        options["true_cfg_scale"] = 1.0
    for _ in range(2 if args.warmup else 1):
        torch.manual_seed(args.seed)
        if hunyuan:
            schedule.set_prompt(args.prompt)
        else:
            schedule.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        video = pipe(**options, latents=latents.clone()).frames[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    from diffusers.utils import export_to_video

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video, str(args.output), fps=24)
    print(f"Saved {args.output}; pipeline {elapsed:.3f}s (text + denoising + VAE).")


if __name__ == "__main__":
    main()
