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

from models import DenoisingDiffusionPipeline
import utils


def _psnr(pred: torch.Tensor, target: torch.Tensor, *, max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """Compute PSNR for tensors in [0, max_val]. Returns a scalar tensor."""
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10((max_val**2) / (mse + eps))


def _gaussian_kernel_2d(*, kernel_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)


def _ssim(pred: torch.Tensor, target: torch.Tensor, *, max_val: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """Compute SSIM for (B,C,H,W) tensors in [0, max_val]. Returns a scalar tensor."""
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("pred/target must be BCHW")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    device = pred.device
    dtype = pred.dtype
    channels = pred.shape[1]

    kernel = _gaussian_kernel_2d(kernel_size=11, sigma=1.5, device=device, dtype=dtype)
    kernel = kernel.repeat(channels, 1, 1, 1)

    mu_x = F.conv2d(pred, kernel, padding=5, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=5, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=5, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=5, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=5, groups=channels) - mu_xy

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps)
    return ssim_map.mean()


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
    _, _, height, width = x.shape
    padded_height_64 = int(64 * ((height + 63) // 64))
    padded_width_64 = int(64 * ((width + 63) // 64))
    x = F.pad(x, (0, padded_width_64 - width, 0, padded_height_64 - height), mode="reflect")
    return x, height, width


def _hann_window_2d(h: int, w: int) -> torch.Tensor:
    window_y = torch.hann_window(h, periodic=False)
    window_x = torch.hann_window(w, periodic=False)
    return (window_y[:, None] * window_x[None, :]).clamp(min=1e-6)


def _blend_window_2d(h: int, w: int, *, power: int) -> torch.Tensor:
    window = _hann_window_2d(h, w)
    if power <= 1:
        return window
    return window**power


@torch.no_grad()
def _infer_tiled(
    diffusion: DenoisingDiffusionPipeline,
    x_cond_cpu: torch.Tensor,
    *,
    tile_sizes: list[int] | None = None,
    overlap: int = 128,
    context: int = 0,
    window_power: int = 1,
) -> torch.Tensor:
    """Overlapping tiled inference. Returns CPU tensor (1,3,H,W)."""

    _, _, height, width = x_cond_cpu.shape

    if tile_sizes is None:
        # Try larger tiles first for speed; fall back on OOM.
        tile_sizes = [1536, 1280, 1024, 768, 512, 384, 256]
    window_cache: dict[tuple[int, int], torch.Tensor] = {}

    if context < 0:
        raise ValueError("context must be >= 0")

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
                for x0 in x_starts:
                    y1 = min(height, y0 + tile_size)
                    x1 = min(width, x0 + tile_size)

                    y0e = max(0, y0 - context)
                    x0e = max(0, x0 - context)
                    y1e = min(height, y1 + context)
                    x1e = min(width, x1 + context)

                    tile_tensor = x_cond_cpu[:, :, y0e:y1e, x0e:x1e].to(diffusion.device)
                    tile_tensor, tile_height, tile_width = _pad_to_64(tile_tensor)

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
                        blend_window = _blend_window_2d(out_h, out_w, power=window_power).unsqueeze(0).unsqueeze(0)
                        window_cache[key] = blend_window

                    output[:, :, y0:y1, x0:x1] += prediction_crop * blend_window
                    weight_map[:, :, y0:y1, x0:x1] += blend_window

            return (output / weight_map).clamp(0, 1)
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
    diffusion = DenoisingDiffusionPipeline(args, config)
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
    def infer_core(
        input_image: Image.Image,
        reference_image: Image.Image | None,
        quality: str,
        blend: str,
        show_stage_outputs: bool,
    ) -> tuple[Image.Image, str, dict]:
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

        conditioning_tensor_cpu = TF.to_tensor(input_image).unsqueeze(0)  # 1x3xHxW in [0,1]
        _, _, height, width = conditioning_tensor_cpu.shape

        debug_lines: list[str] = []
        if show_stage_outputs:
            def _tensor_stats(x: torch.Tensor) -> str:
                x = x.detach()
                # Keep stats lightweight and readable.
                return (
                    f"shape={tuple(x.shape)} dtype={x.dtype} "
                    f"min={x.min().item():.4f} max={x.max().item():.4f} "
                    f"mean={x.mean().item():.4f} std={x.std(unbiased=False).item():.4f}"
                )

            debug_lines.append(f"device={device}")
            debug_lines.append(f"input: shape={(1, 3, height, width)} range=[0,1]")

        try:
            conditioning_tensor = conditioning_tensor_cpu.to(device)
            conditioning_tensor, _, _ = _pad_to_64(conditioning_tensor)

            # Model expects 6-channel input (low + high). For inference we reuse low as both.
            model_input = torch.cat([conditioning_tensor, conditioning_tensor], dim=1)

            if show_stage_outputs:
                # ------------------------
                # Model 1 (stage1 / CTDN) is invoked first to compute low-level features + Retinex pieces.
                # Model 2 (stage2 / diffusion) then predicts a refined feature tensor.
                # Finally, Model 1 is invoked again in decode mode to map predicted features -> RGB.
                # ------------------------
                decom_out = diffusion.model.decom(model_input, pred_fea=None)
                low_fea = decom_out["low_fea"]
                low_R = decom_out["low_R"]
                low_L = decom_out["low_L"]

                debug_lines.append("MODEL1 outputs (from DecompositionReconstructionNet)")
                debug_lines.append(f"- low_fea: {_tensor_stats(low_fea)}")
                debug_lines.append(f"- low_R  : {_tensor_stats(low_R)}")
                debug_lines.append(f"- low_L  : {_tensor_stats(low_L)}")

                # Stage2 diffusion operates in [-1,1] feature space.
                betas = diffusion.model.betas.to(device)
                cond = utils.data_transform(low_fea)
                pred_fea_norm = diffusion.model.sample_training(cond, betas)
                pred_fea = utils.inverse_data_transform(pred_fea_norm)

                debug_lines.append("MODEL2 outputs (from diffusion sampler)")
                debug_lines.append(f"- cond (data_transform(low_fea)): {_tensor_stats(cond)}")
                debug_lines.append(f"- pred_fea_norm ([-1,1]): {_tensor_stats(pred_fea_norm)}")
                debug_lines.append(f"- pred_fea ([0,1]): {_tensor_stats(pred_fea)}")

                pred_img = diffusion.model.decom(model_input, pred_fea=pred_fea)["pred_img"]
            else:
                pred_img = diffusion.model(model_input)["pred_x"]

            predicted_tensor = pred_img[0, :, :height, :width].detach().cpu().clamp(0, 1)
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            print(f"UI: CUDA OOM on full-res {height}x{width}; using tiled inference...", flush=True)
            if show_stage_outputs:
                debug_lines.append(
                    "NOTE: CUDA OOM triggered tiled inference; intermediate stage tensors are not captured in this mode."
                )
            predicted_tensor = _infer_tiled(
                diffusion,
                conditioning_tensor_cpu,
                tile_sizes=tile_sizes,
                overlap=overlap,
                context=context,
                window_power=window_power,
            )[0, :, :, :]

        output_image = TF.to_pil_image(predicted_tensor)

        # Optional metrics (requires a reference/GT image).
        metrics_text = "PSNR/SSIM: N/A (no reference image uploaded)"
        if reference_image is not None:
            ref_img = ImageOps.exif_transpose(reference_image).convert("RGB")
            if ref_img.size != output_image.size:
                # Keep it simple for UI usage: resize ref to match output.
                ref_img = ref_img.resize(output_image.size, Image.BICUBIC)
                resize_note = " (ref resized to match output)"
            else:
                resize_note = ""

            gt = TF.to_tensor(ref_img).unsqueeze(0)  # 1x3xHxW in [0,1]
            pred = predicted_tensor.unsqueeze(0).clamp(0, 1)  # 1x3xHxW
            psnr_val = _psnr(pred, gt).item()
            ssim_val = _ssim(pred, gt).item()
            metrics_text = f"PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}{resize_note}"

        out_dir = os.path.join("results_ui")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        out_path = os.path.join(out_dir, f"enhanced_{ts}_{suffix}.png")
        output_image.save(out_path)
        print(f"UI: saved {out_path}", flush=True)

        if show_stage_outputs:
            debug_text = "\n".join(debug_lines)
            return output_image, metrics_text, gr.update(value=debug_text, visible=True)

        return output_image, metrics_text, gr.update(value="", visible=False)

    @torch.no_grad()
    def infer_enhance(
        input_image: Image.Image,
        quality: str,
        blend: str,
        show_stage_outputs: bool,
    ) -> tuple[Image.Image, dict]:
        output_image, _metrics_text, stage_update = infer_core(
            input_image=input_image,
            reference_image=None,
            quality=quality,
            blend=blend,
            show_stage_outputs=show_stage_outputs,
        )
        return output_image, stage_update

    @torch.no_grad()
    def infer_metrics(
        input_image: Image.Image,
        reference_image: Image.Image,
        quality: str,
        blend: str,
    ) -> tuple[Image.Image, str]:
        output_image, metrics_text, _stage_update = infer_core(
            input_image=input_image,
            reference_image=reference_image,
            quality=quality,
            blend=blend,
            show_stage_outputs=False,
        )
        return output_image, metrics_text

    return infer_enhance, infer_metrics


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

    infer_enhance_fn, infer_metrics_fn = make_infer_fn(args.config, args.resume)

    with gr.Blocks() as demo:
        gr.Markdown("# LLIV stage2 inference")

        with gr.Tabs():
            with gr.TabItem("Enhance"):
                gr.Markdown("Upload a low-light image and enhance it. Enable stage outputs if you want Model1/Model2 intermediate tensor stats.")

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Upload image")

                        with gr.Row():
                            quality = gr.Dropdown(
                                choices=["Low (fast)", "Normal", "Higher quality"],
                                value="Normal",
                                label="Quality",
                            )
                            blend = gr.Dropdown(
                                choices=["High blend (seamless)", "Normal"],
                                value="High blend (seamless)",
                                label="Blend",
                            )

                        show_stage_outputs = gr.Checkbox(
                            value=False,
                            label="Show stage outputs (Model1/Model2)",
                        )
                        run_btn = gr.Button("Enhance")

                    with gr.Column(scale=1):
                        output_image = gr.Image(type="pil", label="Enhanced output")
                        stage_text = gr.Textbox(
                            label="Stage outputs (Model1/Model2)",
                            lines=18,
                            visible=False,
                        )

                run_btn.click(
                    fn=infer_enhance_fn,
                    inputs=[input_image, quality, blend, show_stage_outputs],
                    outputs=[output_image, stage_text],
                )

            with gr.TabItem("Metrics"):
                gr.Markdown("Upload the same scene's reference/GT image to compute PSNR/SSIM against the enhanced output.")

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            input_image_m = gr.Image(type="pil", label="Upload image")
                            reference_image_m = gr.Image(type="pil", label="Reference (GT) image")

                        with gr.Row():
                            quality_m = gr.Dropdown(
                                choices=["Low (fast)", "Normal", "Higher quality"],
                                value="Normal",
                                label="Quality",
                            )
                            blend_m = gr.Dropdown(
                                choices=["High blend (seamless)", "Normal"],
                                value="High blend (seamless)",
                                label="Blend",
                            )

                        run_btn_m = gr.Button("Enhance + Compute metrics")

                    with gr.Column(scale=1):
                        output_image_m = gr.Image(type="pil", label="Enhanced output")
                        metrics_box_m = gr.Textbox(label="Metrics (PSNR/SSIM)", lines=1)

                run_btn_m.click(
                    fn=infer_metrics_fn,
                    inputs=[input_image_m, reference_image_m, quality_m, blend_m],
                    outputs=[output_image_m, metrics_box_m],
                )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
