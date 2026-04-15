import argparse
import os

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


def pad_to_64(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    img_h_64 = int(64 * ((h + 63) // 64))
    img_w_64 = int(64 * ((w + 63) // 64))
    x = F.pad(x, (0, img_w_64 - w, 0, img_h_64 - h), mode="reflect")
    return x, h, w


def _hann_window_2d(h: int, w: int) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False)
    wx = torch.hann_window(w, periodic=False)
    return ((wy[:, None] * wx[None, :]).clamp(min=1e-6)) ** 2


@torch.no_grad()
def infer_tiled(
    *,
    diffusion: DenoisingDiffusion,
    x_cond_cpu: torch.Tensor,
    tile_sizes: list[int] | None = None,
    overlap: int = 128,
) -> torch.Tensor:
    """Run overlapping tiled inference on GPU and return CPU tensor (1,3,H,W)."""

    if tile_sizes is None:
        # Try larger tiles first for speed; fall back on OOM.
        # Keep sizes multiples of 64 to reduce extra padding.
        tile_sizes = [1536, 1280, 1024, 768, 512, 384, 256]

    _, _, H, W = x_cond_cpu.shape
    window_cache: dict[tuple[int, int], torch.Tensor] = {}

    context = max(64, overlap // 2)

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
                print(f"tiled inference: y={y0}/{H}", flush=True)
                for x0 in xs:
                    y1 = min(H, y0 + tile_size)
                    x1 = min(W, x0 + tile_size)

                    y0e = max(0, y0 - context)
                    x0e = max(0, x0 - context)
                    y1e = min(H, y1 + context)
                    x1e = min(W, x1 + context)

                    tile = x_cond_cpu[:, :, y0e:y1e, x0e:x1e].to(diffusion.device)
                    tile, th, tw = pad_to_64(tile)

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
                        win = _hann_window_2d(out_h, out_w).unsqueeze(0).unsqueeze(0)
                        window_cache[key] = win

                    out[:, :, y0:y1, x0:x1] += pred_crop * win
                    weight[:, :, y0:y1, x0:x1] += win

            out = (out / weight).clamp(0, 1)
            return out
        except torch.OutOfMemoryError:
            if diffusion.device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            print(f"Still OOM with tile_size={tile_size}; trying smaller tiles...", flush=True)

    raise torch.OutOfMemoryError("Unable to run tiled inference within GPU memory limits.")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="LLIV stage2 single-image inference")
    parser.add_argument("--config", default="configs/custom_eval.yml", type=str)
    parser.add_argument("--resume", default="ckpt/stage2/stage2_weight.pth.tar", type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", default="results/cli/output.png", type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    cfg = yaml.safe_load(open(args.config, "r"))
    config = dict2namespace(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.resume}")

    ddm_args = argparse.Namespace(mode="evaluation", resume=args.resume, image_folder="results/")
    diffusion = DenoisingDiffusion(ddm_args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=False)
    diffusion.model.eval()

    # Load image with EXIF orientation honored (important for phone photos)
    img = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    x_cond_cpu = TF.to_tensor(img).unsqueeze(0)  # 1x3xHxW in [0,1]
    _, _, h, w = x_cond_cpu.shape

    try:
        x_cond = x_cond_cpu.to(device)
        x_cond, _, _ = pad_to_64(x_cond)
        model_in = torch.cat([x_cond, x_cond], dim=1)  # 1x6xH'xW'
        pred = diffusion.model(model_in)["pred_x"][0, :, :h, :w].detach().cpu().clamp(0, 1)
    except torch.OutOfMemoryError:
        if device.type != "cuda":
            raise
        torch.cuda.empty_cache()
        print(f"CUDA OOM on full-res {h}x{w}; falling back to tiled inference...", flush=True)
        pred = infer_tiled(diffusion=diffusion, x_cond_cpu=x_cond_cpu)[0, :, :, :].clamp(0, 1)

    out_img = TF.to_pil_image(pred)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_img.save(args.output)

    print(f"Saved: {args.output}")
    print(f"Output size: {out_img.size}")


if __name__ == "__main__":
    main()
