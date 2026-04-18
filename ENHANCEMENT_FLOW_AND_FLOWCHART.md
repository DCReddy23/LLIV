# LLIV Enhancement Flow (End-to-End) + Flowchart

This note explains what happens when you give an image to LLIV for enhancement (single image inference, UI inference, or evaluation restoration).

## 1) Main entry points (where enhancement is triggered)

All entry points ultimately call the same diffusion model forward pass and return an enhanced RGB image.

### A) UI

- File: `app.py`
- You upload an image and click a button.
- The UI prepares the tensor, calls the model, then saves a PNG in `results_ui/`.

Key functions involved (UI):

- `app.py: main()`
- `app.py: load_pipeline(config_path, resume_path)`
- `app.py: make_infer_fn(config_path, resume_path)`
- `app.py: infer_core(input_image, reference_image, quality, blend, show_stage_outputs)`
- `app.py: infer_enhance(...)` (Enhance tab wrapper)
- `app.py: infer_metrics(...)` (Metrics tab wrapper)

### B) CLI (single image)

- File: `cli_infer.py`
- You pass `--input` and `--output`.
- Script loads the image, runs inference, saves `--output`.

Key functions involved (CLI):

- `cli_infer.py: main()`
- `cli_infer.py: pad_to_64(x)`
- `cli_infer.py: infer_tiled(diffusion, x_cond_cpu, ...)` (OOM fallback)

### C) Evaluation / restoration loop

- Files: `evaluate.py` → `models/restoration.py`
- It iterates over a `val_loader`, enhances each item, saves results under:
  - `results/<val_dataset>/...`
- If the dataset provides GT (paired data), it can also compute PSNR/SSIM.

Key functions involved (evaluation/restoration):

- `evaluate.py: main()`
- `models/restoration.py: DiffusionRestorationPipeline.__init__(diffusion, args, config)`
- `models/restoration.py: DiffusionRestorationPipeline.restore(val_loader)`
- `models/restoration.py: DiffusionRestorationPipeline._restore_tiled(x_cond_cpu, h, w)` (OOM fallback)

## 2) Inputs and tensor conventions

### A) Image loading

- Image is loaded via PIL and oriented with EXIF correction (important for phone images).
- Then converted to RGB.

### B) Tensor range and shape

- The image becomes a float tensor in **[0, 1]**.
- Shape is **BCHW**:
  - `(B=1, C=3, H, W)` for single-image inference.

### C) Why the model expects 6 channels

Inside the model, Stage1 (`DecompositionReconstructionNet` / `CTDN`) is written to accept a **6-channel** input:

- First 3 channels: low image RGB
- Last 3 channels: high/reference image RGB

But during inference (UI/CLI) you usually **don’t have a paired GT/high image**, so the code feeds:

- `(low, low)` concatenated → `torch.cat([low, low], dim=1)`
- Final input shape becomes `(1, 6, H, W)` (or `(1, 6, H', W')` after padding)

## 3) Preprocessing before model forward

### A) Pad to multiple of 64

Most inference paths pad H/W up to a multiple of 64 using **reflect padding**.

Reason: the network uses downsampling/upsampling stages that work cleanly when spatial sizes are divisible by 64.

- Input `(1, 3, H, W)` becomes `(1, 3, H', W')`
- Then concatenated to `(1, 6, H', W')`

Where in code:

- UI: `app.py: _pad_to_64(x)`
- CLI: `cli_infer.py: pad_to_64(x)`
- Eval: `models/restoration.py: DiffusionRestorationPipeline.restore()` (pads inline with `F.pad(..., 'reflect')`)

### B) Full-resolution vs tiled mode (OOM fallback)

All inference paths try:

1. **Full-resolution inference** first (fast)
2. If CUDA runs out of memory, **tiled inference** (slower but fits in VRAM)

Tiled inference:

- Splits image into overlapping tiles
- Runs the model on each tile
- Blends tiles back together using a Hann window weighting map to reduce seams

## 4) The core model: Stage1 → Stage2 → Stage1

The main logic is in:

- `models/ddm.py`: `LatentRetinexDiffusionModel.forward()` (eval path)
- `models/decom.py`: `DecompositionReconstructionNet.forward()`

### Step 1 — Stage1 (Model1 / CTDN) in _decompose_ mode

Call:

- `decom_output = decom(inputs, pred_fea=None)`

Where in code:

- Called from: `models/ddm.py: LatentRetinexDiffusionModel.forward()` (eval branch)
- Implementation: `models/decom.py: DecompositionReconstructionNet.forward(images, pred_fea=None)`

Outputs (key ones):

- `low_fea`: a compact feature map derived from the low image
- Also Retinex-like pieces in the returned dict (often used for analysis/debug):
  - `low_R` (reflectance), `low_L` (illumination)

Important:

- `low_fea` is the _conditioning_ input to the diffusion model.

### Step 2 — Stage2 (Model2 / diffusion) predicts improved features

Diffusion does not directly generate RGB. It generates a refined feature tensor:

1. Normalize features from [0,1] to [-1,1]

- `cond = data_transform(low_fea)`
- `data_transform(X) = 2X - 1`

Where in code:

- Called from: `models/ddm.py: LatentRetinexDiffusionModel.forward()`
- Implementation: `utils/sampling.py: data_transform(X)` (re-exported as `utils.data_transform`)

2. DDIM-like sampling

- `pred_fea_norm = sample_training(cond, betas)`

Where in code:

- Called from: `models/ddm.py: LatentRetinexDiffusionModel.forward()`
- Implementation: `models/ddm.py: LatentRetinexDiffusionModel.sample_training(x_cond, b, eta=0.)`

3. Map back to [0,1]

- `pred_fea = inverse_data_transform(pred_fea_norm)`
- `inverse_data_transform(X) = clamp((X+1)/2, 0, 1)`

Where in code:

- Called from: `models/ddm.py: LatentRetinexDiffusionModel.forward()`
- Implementation: `utils/sampling.py: inverse_data_transform(X)` (re-exported as `utils.inverse_data_transform`)

### Step 3 — Stage1 again (Model1 / CTDN) in _decode_ mode

Call:

- `pred_img = decom(inputs, pred_fea=pred_fea)["pred_img"]`

Where in code:

- Called from: `models/ddm.py: LatentRetinexDiffusionModel.forward()` (eval branch)
- Implementation: `models/decom.py: DecompositionReconstructionNet.forward(images, pred_fea=...)`

This decodes the predicted features back into an enhanced RGB image.

### Step 4 — Crop back to original size

If padding was applied, the output is cropped back to `(H, W)`.

## 5) Postprocessing and saving

- Output is clamped to [0, 1]
- Converted back to PIL
- Saved to disk:
  - UI: `results_ui/`
  - CLI: `--output`
  - Eval: `results/<val_dataset>/`

Where in code:

- UI: `app.py: infer_core(...)` uses `PIL.Image.save(...)`
- CLI: `cli_infer.py: main()` uses `PIL.Image.save(...)`
- Eval: `models/restoration.py: DiffusionRestorationPipeline.restore()` uses `utils.logging.save_image(...)`
- Save helper: `utils/logging.py: save_image(img, file_directory)`

## 6) Optional metrics (PSNR/SSIM)

### A) Evaluation pipeline

In `models/restoration.py`, if the batch tensor `x` has at least 6 channels:

- `x[:, :3]` is treated as low input
- `x[:, 3:6]` is treated as GT/high reference

Then PSNR/SSIM are computed between:

- `pred_x` (enhanced output)
- `gt` (reference)

Where in code:

- `models/restoration.py: DiffusionRestorationPipeline.restore()`
- Metric helpers: `models/restoration.py: _psnr(...)`, `models/restoration.py: _ssim(...)`

### B) UI metrics tab

In `app.py` metrics mode, PSNR/SSIM are computed only if you upload a reference image.

Where in code:

- `app.py: infer_core(...)` (only when `reference_image is not None`)
- Metric helpers: `app.py: _psnr(...)`, `app.py: _ssim(...)`

## 7) Mermaid flowchart (full flow)

```mermaid
flowchart TD
  A[Input image (PIL)\napp.py: infer_core / cli_infer.py: main] --> B[EXIF transpose + RGB\nPIL.ImageOps.exif_transpose]
  B --> C[To tensor (1,3,H,W) in [0,1]\nTF.to_tensor]
  C --> D[Pad to 64-multiple\napp.py: _pad_to_64 / cli_infer.py: pad_to_64]
  D --> E[Build 6ch input\ntorch.cat(low, low)]

  E --> F{Try full-res inference}
  F -->|Fits GPU| G[Model forward\nmodels/ddm.py: LatentRetinexDiffusionModel.forward]
  F -->|CUDA OOM| T[Tiled inference\napp.py: _infer_tiled / cli_infer.py: infer_tiled / restoration.py: _restore_tiled]

  %% Full model forward
  G --> M1a[Stage1 decompose\nmodels/decom.py: DecompositionReconstructionNet.forward(pred_fea=None)]
  M1a --> LF[Get low_fea (+ low_R, low_L)]
  LF --> N1[data_transform\nutils/sampling.py: data_transform]
  N1 --> M2[Stage2 sample\nmodels/ddm.py: sample_training]
  M2 --> N2[inverse_data_transform\nutils/sampling.py: inverse_data_transform]
  N2 --> M1b[Stage1 decode\nmodels/decom.py: DecompositionReconstructionNet.forward(pred_fea=...)]
  M1b --> O[Crop back to (H,W), clamp to [0,1]]

  %% Tiled inference flow (high-level)
  T --> T1[Split into overlapping tiles + context]
  T1 --> T2[Run same model forward per tile]
  T2 --> T3[Blend tiles with Hann window]
  T3 --> O

  O --> S[Save enhanced image\nUI/CLI: PIL.Image.save | Eval: utils.logging.save_image]

  O --> K{Reference available?}
  K -->|Yes| P[Compute PSNR/SSIM\napp.py: _psnr/_ssim | restoration.py: _psnr/_ssim]
  K -->|No| Q[Metrics N/A]
```

## 8) Quick mental model (1 sentence)

**Stage1 extracts a low-light feature representation, Stage2 diffusion refines that representation, and Stage1 decodes the refined features back into the enhanced RGB image (with padding/tiling helpers to handle large resolutions).**
