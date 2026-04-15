import argparse
import os
import uuid
from datetime import datetime
from functools import lru_cache

import gradio as gr
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from models import DenoisingDiffusion


def dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def _pad_to_64(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pad BCHW tensor with reflect padding to multiples of 64."""
    _, _, h, w = x.shape
    img_h_64 = int(64 * ((h + 63) // 64))
    img_w_64 = int(64 * ((w + 63) // 64))
    x = F.pad(x, (0, img_w_64 - w, 0, img_h_64 - h), mode="reflect")
    return x, h, w


def _hann_window_2d(h: int, w: int) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False)
    wx = torch.hann_window(w, periodic=False)
    return (wy[:, None] * wx[None, :]).clamp(min=1e-6)


def _blend_window_2d(h: int, w: int, *, power: int) -> torch.Tensor:
    win = _hann_window_2d(h, w)
    if power <= 1:
        return win
    return win**power


@torch.no_grad()
def _infer_tiled(
    diffusion: DenoisingDiffusion,
    x_cond_cpu: torch.Tensor,
    *,
    tile_sizes: list[int] | None = None,
    overlap: int = 128,
    context: int = 0,
    window_power: int = 1,
) -> torch.Tensor:
    """Overlapping tiled inference. Returns CPU tensor (1,3,H,W)."""

    _, _, H, W = x_cond_cpu.shape

    if tile_sizes is None:
        # Try larger tiles first for speed; fall back on OOM.
        tile_sizes = [1536, 1280, 1024, 768, 512, 384, 256]
    window_cache: dict[tuple[int, int], torch.Tensor] = {}

    if context < 0:
        raise ValueError("context must be >= 0")

    for tile_size in tile_sizes:
        stride = max(64, tile_size - overlap)
        try:
            out = torch.zeros((1, 3, H, W), dtype=torch.float32)
            weight = torch.zeros((1, 1, H, W), dtype=torch.float32)

            ys = list(range(0, H, stride))
            xs = list(range(0, W, stride))
            if ys[-1] + tile_size < H:
                ys.append(H - tile_size)
            if xs[-1] + tile_size < W:
                xs.append(W - tile_size)
            ys = sorted(set(max(0, y0) for y0 in ys))
            xs = sorted(set(max(0, x0) for x0 in xs))

            for y0 in ys:
                for x0 in xs:
                    y1 = min(H, y0 + tile_size)
                    x1 = min(W, x0 + tile_size)

                    y0e = max(0, y0 - context)
                    x0e = max(0, x0 - context)
                    y1e = min(H, y1 + context)
                    x1e = min(W, x1 + context)

                    tile = x_cond_cpu[:, :, y0e:y1e, x0e:x1e].to(diffusion.device)
                    tile, th, tw = _pad_to_64(tile)

                    pred = diffusion.model(torch.cat([tile, tile], dim=1))["pred_x"][:, :, :th, :tw]
                    pred_cpu = pred.detach().cpu()

                    cy0 = y0 - y0e
                    cx0 = x0 - x0e
                    cy1 = cy0 + (y1 - y0)
                    cx1 = cx0 + (x1 - x0)
                    pred_crop = pred_cpu[:, :, cy0:cy1, cx0:cx1]

                    out_h = y1 - y0
                    out_w = x1 - x0
                    key = (out_h, out_w)
                    win = window_cache.get(key)
                    if win is None:
                        win = _blend_window_2d(out_h, out_w, power=window_power).unsqueeze(0).unsqueeze(0)
                        window_cache[key] = win

                    out[:, :, y0:y1, x0:x1] += pred_crop * win
                    weight[:, :, y0:y1, x0:x1] += win

            return (out / weight).clamp(0, 1)
        except torch.OutOfMemoryError:
            if diffusion.device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            print(f"UI: still OOM with tile_size={tile_size}; trying smaller tiles...", flush=True)

    raise torch.OutOfMemoryError("UI: unable to run tiled inference within GPU memory limits")


@lru_cache(maxsize=1)
def load_pipeline(config_path: str, resume_path: str):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    print(f"Loading config: {config_path}", flush=True)
    with open(config_path, "r") as f:
        config = dict2namespace(yaml.safe_load(f))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device

    print(f"Using device: {device}", flush=True)

    args = argparse.Namespace(mode="evaluation", resume=resume_path, image_folder="results/")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(resume_path, ema=False)
    diffusion.model.eval()

    print(f"Loaded checkpoint: {resume_path}", flush=True)

    return diffusion, device


def make_infer_fn(config_path: str, resume_path: str):
    diffusion, device = load_pipeline(config_path, resume_path)

    low_steps = 10
    normal_steps = int(getattr(diffusion.config.diffusion, "num_sampling_timesteps", 20))
    quality_steps = 50

    @torch.no_grad()
    def infer(input_image: Image.Image, quality: str, blend: str) -> Image.Image:
        if input_image is None:
            raise gr.Error("Please upload an image")

        if quality == "Low (fast)":
            steps = low_steps
        elif quality == "Higher quality":
            steps = quality_steps
        else:
            steps = normal_steps

        if steps <= 0 or steps > int(diffusion.config.diffusion.num_diffusion_timesteps):
            raise gr.Error(
                f"Invalid sampling steps: {steps}. Must be within 1..{diffusion.config.diffusion.num_diffusion_timesteps}"
            )

        diffusion.config.diffusion.num_sampling_timesteps = steps
        diffusion.model.config.diffusion.num_sampling_timesteps = steps

        # Heuristics for 4GB-class GPUs on large photos (e.g., 6000x4000):
        # - More sampling steps => more total work, and larger tiles are more likely to OOM.
        # - Blend mode controls seam reduction vs speed.
        if steps <= 10:
            tile_sizes = [1536, 1280, 1024, 768, 512, 384, 256]
        elif steps <= 20:
            tile_sizes = [1280, 1024, 768, 512, 384, 256]
        else:
            tile_sizes = [768, 640, 512, 384, 256]

        if blend == "High blend (seamless)":
            # More overlap + extra context + stronger center weighting.
            if steps <= 10:
                overlap = 96
            elif steps <= 20:
                overlap = 160
            else:
                overlap = 256
            context = max(64, overlap // 2)
            window_power = 2
        else:
            # Older/faster blend: less overlap, no context, standard Hann window.
            if steps <= 10:
                overlap = 64
            elif steps <= 20:
                overlap = 96
            else:
                overlap = 128
            context = 0
            window_power = 1

        # Fix common phone-photo orientation issues.
        input_image = ImageOps.exif_transpose(input_image).convert("RGB")

        x_cond_cpu = TF.to_tensor(input_image).unsqueeze(0)  # 1x3xHxW in [0,1]
        _, _, h, w = x_cond_cpu.shape

        try:
            x_cond = x_cond_cpu.to(device)
            x_cond, _, _ = _pad_to_64(x_cond)

            # Model expects 6-channel input (low + high). For inference we reuse low as both.
            model_in = torch.cat([x_cond, x_cond], dim=1)
            pred_x = diffusion.model(model_in)["pred_x"][0, :, :h, :w].detach().cpu().clamp(0, 1)
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            print(f"UI: CUDA OOM on full-res {h}x{w}; using tiled inference...", flush=True)
            pred_x = _infer_tiled(
                diffusion,
                x_cond_cpu,
                tile_sizes=tile_sizes,
                overlap=overlap,
                context=context,
                window_power=window_power,
            )[0, :, :, :]

        out_img = TF.to_pil_image(pred_x)

        out_dir = os.path.join("results_ui")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        out_path = os.path.join(out_dir, f"enhanced_{ts}_{suffix}.png")
        out_img.save(out_path)
        print(f"UI: saved {out_path}", flush=True)

        return out_img

    return infer


def main():
    parser = argparse.ArgumentParser(description="LLIV stage2 inference UI")
    parser.add_argument("--config", default="configs/custom_eval.yml", type=str)
    parser.add_argument("--resume", default="ckpt/stage2/stage2_weight.pth.tar", type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    print("Starting LLIV stage2 UI...", flush=True)
    print(f"- config: {args.config}", flush=True)
    print(f"- resume: {args.resume}", flush=True)
    print(f"- url: http://{args.host}:{args.port}", flush=True)

    infer_fn = make_infer_fn(args.config, args.resume)

    demo = gr.Interface(
        fn=infer_fn,
        inputs=[
            gr.Image(type="pil", label="Upload image"),
            gr.Dropdown(
                choices=["Low (fast)", "Normal", "Higher quality"],
                value="Normal",
                label="Quality",
            ),
            gr.Dropdown(
                choices=["High blend (seamless)", "Normal"],
                value="High blend (seamless)",
                label="Blend",
            ),
        ],
        outputs=gr.Image(type="pil", label="Enhanced output"),
    )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
