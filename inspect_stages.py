import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from models import DenoisingDiffusionPipeline
import utils


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


def _stats(name: str, x: torch.Tensor) -> str:
    x_detached = x.detach()
    return (
        f"{name}: shape={tuple(x_detached.shape)} dtype={x_detached.dtype} "
        f"min={x_detached.min().item():.4f} max={x_detached.max().item():.4f} "
        f"mean={x_detached.mean().item():.4f} std={x_detached.std(unbiased=False).item():.4f}"
    )


def _save_tensor_as_image(path: str, x: torch.Tensor) -> None:
    # x: (3,H,W) in [0,1]
    x = x.detach().clamp(0, 1)
    img = TF.to_pil_image(x)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LLIV stage1/model1 and stage2/model2 outputs")
    parser.add_argument("--config", default="configs/custom_eval.yml")
    parser.add_argument("--resume", default="ckpt/stage2/stage2_weight.pth.tar")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="results_gpu/inspect")
    args = parser.parse_args()

    raw_config = yaml.safe_load(open(args.config, "r"))
    config = dict2namespace(raw_config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device

    print(f"Using device: {device}")

    ddm_args = SimpleNamespace(mode="evaluation", resume=args.resume, image_folder="results/")
    diffusion = DenoisingDiffusionPipeline(ddm_args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=False)
    diffusion.model.eval()

    input_image = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    x_rgb_cpu = TF.to_tensor(input_image).unsqueeze(0)  # 1x3xHxW in [0,1]
    x_rgb = x_rgb_cpu.to(device)

    x_rgb_pad, orig_h, orig_w = pad_to_64(x_rgb)
    x6 = torch.cat([x_rgb_pad, x_rgb_pad], dim=1)  # 1x6xH'xW'

    print(_stats("input_rgb_padded", x_rgb_pad))

    # ------------------ Model 1 (stage1 / decom) ------------------
    decom = diffusion.model.decom
    decom_out = decom(x6, pred_fea=None)

    # These are the direct "Model1 outputs" used by stage2.
    low_fea = decom_out["low_fea"]
    low_R = decom_out["low_R"]
    low_L = decom_out["low_L"]

    print("\nMODEL1 (CTDN / DecompositionReconstructionNet) outputs")
    print(_stats("low_fea (H/8,W/8)", low_fea))
    print(_stats("low_R   (H/8,W/8)", low_R))
    print(_stats("low_L   (H/8,W/8)", low_L))

    # Save low-res maps upsampled for viewing
    up = torch.nn.functional.interpolate
    low_fea_up = up(low_fea, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0]
    low_R_up = up(low_R, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0]
    low_L_up = up(low_L, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0]

    _save_tensor_as_image(os.path.join(args.outdir, "model1_low_fea_up.png"), low_fea_up)
    _save_tensor_as_image(os.path.join(args.outdir, "model1_low_R_up.png"), low_R_up)
    _save_tensor_as_image(os.path.join(args.outdir, "model1_low_L_up.png"), low_L_up)

    # ------------------ Model 2 (stage2 / diffusion) ------------------
    betas = diffusion.model.betas.to(device)
    cond = utils.data_transform(low_fea)  # [-1,1]

    pred_fea_norm = diffusion.model.sample_training(cond, betas)
    pred_fea = utils.inverse_data_transform(pred_fea_norm)  # back to [0,1]

    print("\nMODEL2 (DiffusionUNet sampler) outputs")
    print(_stats("cond = data_transform(low_fea)", cond))
    print(_stats("pred_fea_norm (model2 output, [-1,1])", pred_fea_norm))
    print(_stats("pred_fea (inverse transform, [0,1])", pred_fea))

    pred_fea_up = up(pred_fea, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0]
    _save_tensor_as_image(os.path.join(args.outdir, "model2_pred_fea_up.png"), pred_fea_up)

    # Decode using Model1 reconstruction path
    pred_img = decom(x6, pred_fea=pred_fea)["pred_img"]
    pred_img = pred_img[:, :, :orig_h, :orig_w].detach().clamp(0, 1)

    print("\nFINAL (Model1 decoder using Model2 features)")
    print(_stats("pred_x (RGB)", pred_img))

    _save_tensor_as_image(os.path.join(args.outdir, "pred_x.png"), pred_img[0])

    print(f"\nSaved inspection images to: {args.outdir}")


if __name__ == "__main__":
    main()
