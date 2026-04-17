import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from models import DenoisingDiffusionPipeline


def dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def pad_to_64(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = x.shape
    padded_height_64 = int(64 * ((height + 63) // 64))
    padded_width_64 = int(64 * ((width + 63) // 64))
    x = F.pad(x, (0, padded_width_64 - width, 0, padded_height_64 - height), mode="reflect")
    return x, height, width


def _hann_window_2d(h: int, w: int) -> torch.Tensor:
    window_y = torch.hann_window(h, periodic=False)
    window_x = torch.hann_window(w, periodic=False)
    return ((window_y[:, None] * window_x[None, :]).clamp(min=1e-6)) ** 2


@torch.no_grad()
def infer_tiled(
    *,
    diffusion: DenoisingDiffusionPipeline,
    x_cond_cpu: torch.Tensor,
    tile_sizes: list[int] | None = None,
    overlap: int = 128,
) -> torch.Tensor:
    """Run overlapping tiled inference on GPU and return CPU tensor (1,3,H,W)."""

    if tile_sizes is None:
        # Try larger tiles first for speed; fall back on OOM.
        # Keep sizes multiples of 64 to reduce extra padding.
        tile_sizes = [1536, 1280, 1024, 768, 512, 384, 256]

    _, _, height, width = x_cond_cpu.shape
    window_cache: dict[tuple[int, int], torch.Tensor] = {}

    context = max(64, overlap // 2)

    for tile_size in tile_sizes:
        stride = max(64, tile_size - overlap)
        try:
            output = torch.zeros((1, 3, height, width), dtype=torch.float32)
            weight_map = torch.zeros((1, 1, height, width), dtype=torch.float32)

            y_starts = list(range(0, height, stride))
            x_starts = list(range(0, width, stride))
            if y_starts[-1] + tile_size < height:
                y_starts.append(height - tile_size)
            if x_starts[-1] + tile_size < width:
                x_starts.append(width - tile_size)
            y_starts = sorted(set(max(0, y0) for y0 in y_starts))
            x_starts = sorted(set(max(0, x0) for x0 in x_starts))

            for y0 in y_starts:
                print(f"tiled inference: y={y0}/{height}", flush=True)
                for x0 in x_starts:
                    y1 = min(height, y0 + tile_size)
                    x1 = min(width, x0 + tile_size)

                    y0e = max(0, y0 - context)
                    x0e = max(0, x0 - context)
                    y1e = min(height, y1 + context)
                    x1e = min(width, x1 + context)

                    tile_tensor = x_cond_cpu[:, :, y0e:y1e, x0e:x1e].to(diffusion.device)
                    tile_tensor, tile_height, tile_width = pad_to_64(tile_tensor)

                    prediction = diffusion.model(torch.cat([tile_tensor, tile_tensor], dim=1))["pred_x"][:, :, :tile_height, :tile_width]
                    prediction_cpu = prediction.detach().cpu()

                    crop_y0 = y0 - y0e
                    crop_x0 = x0 - x0e
                    crop_y1 = crop_y0 + (y1 - y0)
                    crop_x1 = crop_x0 + (x1 - x0)
                    prediction_crop = prediction_cpu[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

                    out_h = y1 - y0
                    out_w = x1 - x0
                    key = (out_h, out_w)
                    blend_window = window_cache.get(key)
                    if blend_window is None:
                        blend_window = _hann_window_2d(out_h, out_w).unsqueeze(0).unsqueeze(0)
                        window_cache[key] = blend_window

                    output[:, :, y0:y1, x0:x1] += prediction_crop * blend_window
                    weight_map[:, :, y0:y1, x0:x1] += blend_window

            output = (output / weight_map).clamp(0, 1)
            return output
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

    raw_config = yaml.safe_load(open(args.config, "r"))
    config = dict2namespace(raw_config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.resume}")

    ddm_args = argparse.Namespace(mode="evaluation", resume=args.resume, image_folder="results/")
    diffusion = DenoisingDiffusionPipeline(ddm_args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=False)
    diffusion.model.eval()

    # Load image with EXIF orientation honored (important for phone photos)
    input_image = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    conditioning_tensor_cpu = TF.to_tensor(input_image).unsqueeze(0)  # 1x3xHxW in [0,1]
    _, _, height, width = conditioning_tensor_cpu.shape

    try:
        conditioning_tensor = conditioning_tensor_cpu.to(device)
        conditioning_tensor, _, _ = pad_to_64(conditioning_tensor)
        model_input = torch.cat([conditioning_tensor, conditioning_tensor], dim=1)  # 1x6xH'xW'
        predicted_tensor = diffusion.model(model_input)["pred_x"][0, :, :height, :width].detach().cpu().clamp(0, 1)
    except torch.OutOfMemoryError:
        if device.type != "cuda":
            raise
        torch.cuda.empty_cache()
        print(f"CUDA OOM on full-res {height}x{width}; falling back to tiled inference...", flush=True)
        predicted_tensor = infer_tiled(diffusion=diffusion, x_cond_cpu=conditioning_tensor_cpu)[0, :, :, :].clamp(0, 1)

    output_image = TF.to_pil_image(predicted_tensor)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_image.save(args.output)

    print(f"Saved: {args.output}")
    print(f"Output size: {output_image.size}")


if __name__ == "__main__":
    main()
