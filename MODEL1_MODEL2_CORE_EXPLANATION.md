# LLIV Core Model Explanation (Stage1/Model1 + Stage2/Model2)

This note explains **only the core model**: what Stage1 (Model1) does, what Stage2 (Model2) does, and **how they pass tensors to each other in code**.

## 1) Core idea (two-stage pipeline)

- **Model 1 (Stage1 / CTDN)** is a _feature extractor + Retinex decomposition + decoder_ implemented as `DecompositionReconstructionNet` in `models/decom.py`.
  - In **decompose mode** it produces a low-resolution feature tensor `low_fea` plus Retinex-style pieces (`low_R`, `low_L`).
  - In **decode mode** it takes a predicted feature tensor `pred_fea` and reconstructs an RGB image `pred_img`.

- **Model 2 (Stage2 / diffusion)** is a _DDIM-like sampler driven by a U-Net_ implemented as `LatentRetinexDiffusionModel` in `models/ddm.py`.
  - It **does not directly enhance pixels** inside the diffusion loop.
  - It predicts **better features** in the same space/shape as `low_fea`, then **Model 1 decodes** those features back to RGB.

## 2) The exact “handshake” between Model 1 and Model 2

### Input to the system

- The model is called with a **6-channel tensor** shaped `(B, 6, H, W)`:
  - `inputs[:, :3, ...]` = low image (RGB)
  - `inputs[:, 3:, ...]` = high/reference image (RGB)
- In inference, since there is no paired “high” image, the code often duplicates low → both halves.

### Model 1 → Model 2

- Model 1 is invoked as:
  - `decom_output = decom(inputs, pred_fea=None)`

- The most important output for Stage2 inference is:
  - `low_fea` with shape `(B, 3, H/8, W/8)`

### Model 2 → Model 1

- Model 2 produces:
  - `pred_fea` with the **same shape** `(B, 3, H/8, W/8)`

- Model 1 is invoked again in decode mode:
  - `pred_img = decom(inputs, pred_fea=pred_fea)["pred_img"]`

## 3) What Model 1 outputs (Stage1 / `DecompositionReconstructionNet`)

Model 1 lives in `models/decom.py` and has **two modes**:

### 3.1 Decompose mode (`pred_fea is None`)

Call:

- `CTDN(x_in, pred_fea=None)`

Returns a dict with keys (core ones):

- `low_fea`: `(B,3,H/8,W/8)` — the feature tensor used to condition diffusion
- `low_R`: `(B,3,H/8,W/8)` — estimated reflectance
- `low_L`: `(B,3,H/8,W/8)` — estimated illumination
- `high_fea`, `high_R`, `high_L`: same outputs computed from the 2nd half of `x_in`

How it computes these:

1. **Feature extraction / pyramid** produces low/high downsampled representations.
2. **Channel reduction** produces a compact 3-channel `low_fea`.
3. **Retinex decomposition** produces `(R, L)` from those features.

### 3.2 Decode mode (`pred_fea is given`)

Call:

- `CTDN(x_in, pred_fea=<tensor>)`

Returns:

- `pred_img`: `(B,3,H,W)` — reconstructed RGB image

## 4) What Model 2 outputs (Stage2 / diffusion)

Model 2 lives in `models/ddm.py`:

- `LatentRetinexDiffusionModel.sample_training(...)` performs DDIM-like sampling.
- `LatentRetinexDiffusionModel.forward(...)` wires Stage1 and Stage2 together.

### 4.1 Feature-space normalization

Diffusion runs in `[-1, 1]`. The repo uses:

- `data_transform(X) = 2X - 1` mapping `[0,1] → [-1,1]`
- `inverse_data_transform(X) = clamp((X+1)/2, 0, 1)` mapping `[-1,1] → [0,1]`

These are in `utils/sampling.py`.

### 4.2 Evaluation (inference) pipeline

In eval mode, Model 2 does:

1. Run Model 1 to get Stage1 features:
   - `low_fea = decom(inputs, pred_fea=None)["low_fea"]`
2. Normalize conditioning:
   - `cond = data_transform(low_fea)`
3. Sample predicted features with diffusion:
   - `pred_fea_norm = sample_training(cond, betas)`
4. Map back to `[0,1]`:
   - `pred_fea = inverse_data_transform(pred_fea_norm)`
5. Decode with Model 1:
   - `pred_x = decom(inputs, pred_fea=pred_fea)["pred_img"]`

So the “real” Stage2 output is **the predicted feature tensor** `pred_fea`; the final image is produced by Stage1’s decoder.

## 5) Training-time relationship (why `low_R`, `low_L`, `high_L` are used)

In training mode, `LatentRetinexDiffusionModel.forward()` also computes targets and losses:

- Stage1 provides `low_R`, `low_L`, `low_fea`, and `high_L`.
- The diffusion U-Net is trained with a **noise-prediction loss** (MSE on predicted noise).
- There is also a small feature consistency term (`scc_loss`) between:
  - `pred_fea` (sampled/predicted)
  - `reference_fea = low_R * (low_L ** 0.2)`

Stage1 (`decom`) parameters are frozen during stage2 training (training loop sets `requires_grad=False` for params containing `"decom"`).

## 6) One-sentence explanation (for presenting)

Model 1 converts a low-light image into a compact, Retinex-informed feature representation and can decode features back to an image; Model 2 uses diffusion to transform Model 1’s low-light features into enhanced features, which Model 1 then decodes into the final enhanced RGB.
