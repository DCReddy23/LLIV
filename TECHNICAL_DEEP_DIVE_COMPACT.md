# LLIV — Compact Technical Brief (for teammates)

This is the **short version** of `TECHNICAL_DEEP_DIVE.md`: enough to understand the repo, run inference, and know where to look in code.

## 1) What it does

- **Input:** a low-light RGB image
- **Output:** an enhanced RGB image
- **Core idea:** a **two-stage pipeline**
  - **Stage1 (DecompositionReconstructionNet / Retinex-like)**: decomposes low/high into reflectance + illumination and provides a decoder to reconstruct an image from “features”. (Alias: `CTDN`)
  - **Stage2 (Diffusion)**: uses a diffusion sampler (DDIM-like) + U-Net to **predict better features**, then the decomposition/reconstruction net reconstructs the enhanced image.

In practice, for enhancement you run **stage2 inference** using:

- `ckpt/stage2/stage2_weight.pth.tar`

## 2) Fast repo map (what matters)

**Run it**

- `cli_infer.py`: one image → one output (recommended for quick usage + saving output)
- `app.py`: Gradio UI upload → preview + auto-save
- `evaluate.py`: runs on a filelist dataset (batch-like evaluation)

**Core model code**

- `models/ddm.py`: diffusion wrapper, sampling, checkpoint loading
- `models/unet.py`: diffusion U-Net backbone
- `models/decom.py`: `DecompositionReconstructionNet` (alias: `CTDN`) decomposition + reconstruction (Retinex-inspired)
- `models/restoration.py`: eval-time restoration + CUDA OOM tiled fallback

**Data + utils**

- `datasets/dataset.py`: loads paired low/high and returns a `(6,H,W)` tensor
- `utils/sampling.py`: range transforms `[0,1] ↔ [-1,1]`
- `utils/logging.py`: checkpoint loading + image saving

## 3) The inference data flow (one picture)

**Notation:** `x_low` is RGB in `[0,1]`, shape `(B,3,H,W)`.

1. Build a 6-channel input by duplicating the low image:
   - `x_in = cat([x_low, x_low], dim=1)` → `(B,6,H,W)`
2. Decomposition/reconstruction net decomposition on `x_in` produces a low-light feature tensor `low_fea` (plus reflectance/illumination pieces internally).
3. Normalize conditioning features for diffusion:
   - `low_condition = data_transform(low_fea)` → `[-1,1]`
4. Diffusion sampling predicts a “better” feature tensor:
   - `pred_fea = sample_training(low_condition, betas)`
5. Map predicted features back to `[0,1]`.
6. Decomposition/reconstruction net reconstruction uses `pred_fea` to decode the enhanced RGB:
   - `pred_img = DecompositionReconstructionNet(x_in, pred_fea=pred_fea)["pred_img"]` (alias: `CTDN`)

### 3.1) What each stage outputs (what to look at)

**Stage1 / Model1 (`DecompositionReconstructionNet`, alias `CTDN`)**

When called as `CTDN(x_in, pred_fea=None)` it returns a dict with:

- `low_fea`: `(B,3,H/8,W/8)` — low-resolution feature tensor used as the diffusion conditioning
- `low_R`: `(B,3,H/8,W/8)` — estimated reflectance
- `low_L`: `(B,3,H/8,W/8)` — estimated illumination
- `high_fea`, `high_R`, `high_L`: same outputs computed from the 2nd half of `x_in` (the “high” image)

When called as `CTDN(x_in, pred_fea=<tensor>)` it returns:

- `pred_img`: `(B,3,H,W)` — reconstructed/enhanced RGB image

**Stage2 / Model2 (diffusion sampler + U-Net)**

- Consumes `cond = data_transform(low_fea)` where `data_transform(X)=2X-1` maps `[0,1] → [-1,1]`
- Produces `pred_fea_norm` in `[-1,1]`, then `pred_fea = inverse_data_transform(pred_fea_norm)` back in `[0,1]`
- `pred_fea` is fed back into Stage1 to decode the final RGB

### 3.2) UI: showing stage outputs during inference

The Gradio UI (`app.py`) now returns two outputs:

- the final enhanced image
- a text panel that prints the stage outputs (Model1 dict keys + tensor stats, and Model2 feature stats)

The stage-output panel is **hidden by default** and can be toggled with:

- `Show stage outputs (Model1/Model2)`

This is useful for verifying what Model1 is producing (`low_fea/low_R/low_L`) and what Model2 is changing (`pred_fea`).

## 4) What controls quality vs speed

The **main knob** is the number of sampling steps:

- `config.diffusion.num_sampling_timesteps`

Rule of thumb:

- 10 steps: much faster, lower quality
- 20 steps: “normal” baseline
- 50 steps: slower, higher quality

In the UI (`app.py`):

- `Low (fast)` sets steps to `10`
- `Normal` uses config (often `20`)
- `Higher quality` sets steps to `50`

## 5) GPU OOM + tiling (important for iPhone photos)

Large images (e.g., ~6000×4000) can exceed VRAM on 4GB GPUs. The repo handles this:

- Full-res inference is tried first.
- If CUDA OOM happens, it falls back to **overlapping tiled inference**:
  - run the model on overlapping tiles
  - blend tile outputs using a Hann window (edges down-weighted)
  - optional **context padding** around each tile (run a bigger tile, crop to core) to reduce seams

Where tiling happens:

- `cli_infer.py` (single image)
- `models/restoration.py` (evaluate path)
- `app.py` (UI)

**Seams vs speed tradeoff:**

- More overlap/context/stronger window → fewer seams but slower
- Bigger tile size (if VRAM allows) → faster (fewer tiles)

UI also exposes this as “Blend”:

- `Normal`: faster tiling
- `High blend (seamless)`: stronger blending + more context (slower, fewer seams)

## 6) Image orientation gotcha (EXIF)

Phone photos may be stored “sideways” and rely on EXIF orientation metadata.

The repo corrects this in inference loaders using `ImageOps.exif_transpose` so the output matches the perceived orientation.

## 7) Where outputs go

- UI auto-saves to `results_ui/` as `enhanced_<timestamp>_<id>.png`
- CLI inference typically saves under its configured output folder (see `cli_infer.py` args)
- `evaluate.py` saves to `<args.image_folder>/<val_dataset>/...`

## 8) If you only read 4 files

1. `models/ddm.py` — how the diffusion wrapper calls the decomposition net + U-Net and samples
2. `models/decom.py` — what the decomposition net is producing/consuming (features in/out)
3. `cli_infer.py` — practical inference (padding, EXIF, OOM fallback)
4. `app.py` — UI knobs (Quality + Blend) and auto-save behavior

## 9) Quick run examples (typical)

CLI (single image):

- Use `cli_infer.py` with your config and stage2 checkpoint.

UI:

- Run `app.py` and upload an image.

Evaluation:

- Prepare `demo_data/<val_dataset>_val.txt` with lines: `low_path high_path`
- Run `evaluate.py` using the chosen config and checkpoint.
