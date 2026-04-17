# LLIV Project Guide (Code Map + How To Run)

This repository contains a two-stage low-light image enhancement pipeline. In practice you will mostly **run stage2 inference** using a provided checkpoint.

If you want a focused explanation of just the core architecture (Model1/Model2), see: `MODEL1_MODEL2_CORE_EXPLANATION.md`.

## 1) Repository map (what each file/folder does)

### Top-level entrypoints

- `train.py`
  - **Purpose:** stage2 training entrypoint.
  - **Reads:** a YAML config from `configs/<name>.yml`.
  - **Creates:** `models.DenoisingDiffusionPipeline` and calls `diffusion.train(dataset)`.
  - **Outputs:** periodic checkpoint files under `config.data.ckpt_dir` (see config) and validation patch images under `--image_folder`.

- `evaluate.py`
  - **Purpose:** dataset-based evaluation/restoration entrypoint.
  - **Reads:** `configs/<name>.yml` + a paired validation filelist `demo_data/<val_dataset>_val.txt` (or `<data_dir>/<val_dataset>_val.txt`).
  - **Runs:** `models.DiffusionRestorationPipeline.restore(val_loader)`.
  - **Outputs:** images saved under `--image_folder/<val_dataset>/...`.

- `cli_infer.py`
  - **Purpose:** single-image inference (no dataset/filelist needed).
  - **Reads:** `--input` image.
  - **Runs:** stage2 model directly.
  - **Outputs:** a single file at `--output`.
  - **Important:** Includes **EXIF orientation fix** and **CUDA OOM → tiled inference fallback** for large phone photos.

- `app.py`
  - **Purpose:** upload UI (Gradio) for inference.
  - **Runs:** stage2 model on uploaded image.
  - **Important:** Includes **EXIF orientation fix** and **CUDA OOM → tiled inference fallback**.
  - **UI controls:**
    - **Quality:** `Low (fast)` / `Normal` / `Higher quality`
    - **Blend:** `Normal` (faster) / `High blend (seamless)` (slower, fewer visible tile seams)
  - **Saves:** auto-saves each output image under `results_ui/`.
  - **Gradio state:** Gradio may create `.gradio/` (e.g. `.gradio/flagged/`).

### Configs

- `configs/unsupervised.yml`
  - Default training-style config (paths likely point to a non-local dataset location).

- `configs/custom_eval.yml`
  - Local eval config that points at `demo_data/` and uses `val_dataset: custom`.

### Datasets

- `datasets/dataset.py`
  - `LLdataset.get_loaders()` builds train/val `DataLoader` objects.
  - `AllWeatherDataset` reads a text file containing **pairs**:
    - Each line format: `path/to/low_image path/to/high_image`
  - `get_images()` returns:
    - `x`: a 6-channel tensor `torch.cat([low_img, high_img], dim=0)` shaped `(6, H, W)`
    - `img_id`: taken from the low image filename
  - **EXIF orientation** is honored via `ImageOps.exif_transpose(...)` so iPhone photos don’t appear rotated/misaligned.

- `datasets/data_augment.py`
  - Pair-wise transforms (random flips, to-tensor) applied consistently to low/high.

### Models

- `models/ddm.py`
  - **Core of stage2**.
  - `LatentRetinexDiffusionModel` (alias: `Net`)
    - Contains:
      - `Unet`: `models.unet.DiffusionUNet`
      - `decom`: `models.decom.DecompositionReconstructionNet` (alias: `CTDN`) (Retinex-like decomposition + reconstruction)
    - In evaluation (`args.mode != 'training'`): the decomposition/reconstruction net is constructed and then **its weights are loaded from the stage2 checkpoint** via `load_ddm_ckpt()`.
    - `forward(inputs)`
      - **Inputs:** typically `(B, 6, H, W)`; evaluation path uses `torch.cat([x_cond, x_cond], dim=1)` so it becomes 6 channels.
      - **Outputs:** a dict. In evaluation mode, it returns `{"pred_x": <enhanced image>}`.
  - `DenoisingDiffusionPipeline` (alias: `DenoisingDiffusion`)
    - Wraps the model, optimizer, EMA, training loop.
    - `load_ddm_ckpt(path, ema=False)` loads `checkpoint['state_dict']` and strips `module.` prefixes if the checkpoint came from `nn.DataParallel`.

- `models/restoration.py`
  - `DiffusionRestorationPipeline` (alias: `DiffusiveRestoration`)
    - Loads the checkpoint from `args.resume` and exposes `restore(val_loader)`.
    - In `restore()`:
      - Takes the dataset tensor `x` (6-channel), uses only the first 3 channels as `x_cond`.
      - Pads to multiples of 64 (reflect padding).
      - Runs the model as `model(torch.cat([x_cond, x_cond], dim=1))["pred_x"]`.
      - Saves output images.
    - **GPU note:** if the full-resolution image causes CUDA OOM, it automatically falls back to **overlapping tiled inference**.

- `models/decom.py`
  - `DecompositionReconstructionNet` (alias: `CTDN`) (decomposition/reconstruction network)
    - `forward(images, pred_fea=None)` returns different outputs:
      - If `pred_fea is None`: returns decomposition features/retinex outputs (`low_R`, `low_L`, `low_fea`, `high_L`, etc.).
      - Else: returns reconstructed image in `output["pred_img"]`.

- `models/unet.py`
  - `DiffusionUNet`: a standard U-Net with timestep embeddings, ResNet blocks, and attention in one resolution.

### Utils

- `utils/sampling.py`
  - `data_transform(X)`: maps `[0,1] → [-1,1]`
  - `inverse_data_transform(X)`: maps `[-1,1] → [0,1]`

- `utils/logging.py`
  - `save_image(img, path)`: uses `torchvision.utils.save_image`.
  - `load_checkpoint(path, device)`: `torch.load(..., map_location=device)` when device is provided.

- `utils/optimize.py`
  - `get_optimizer(config, parameters)`: builds optimizer (Adam/RMSProp/SGD).

## 2) Where do I put inputs? Where do outputs go?

### A) Single image (recommended)

- Input: any image file path you pass to `cli_infer.py --input ...`
- Output: exactly the file path you pass via `--output ...`

### B) Dataset/filelist eval (`evaluate.py`)

- Inputs:
  - A YAML config specifying:
    - `data.data_dir`
    - `data.val_dataset`
  - A filelist at: `<data_dir>/<val_dataset>_val.txt`
    - Each line: `low_path high_path`
- Outputs:
  - Folder: `--image_folder/<val_dataset>/`

## 3) Checkpoints (stage1 vs stage2)

Expected checkpoint paths in this workspace:

- `ckpt/stage1/stage1_weight.pth.tar`
  - Used when training stage2 (stage1 is loaded/frozen in `models/ddm.py` during training).

- `ckpt/stage2/stage2_weight.pth.tar`
  - Used for inference/evaluation.
  - Contains the weights needed for stage2 inference (including the decomposition/reconstruction weights as part of the model state_dict).

## Note on names (backwards compatibility)

This repo recently renamed several classes for readability. The old names still work as aliases, so external code won’t break:

- `CTDN` → `DecompositionReconstructionNet`
- `Net` → `LatentRetinexDiffusionModel`
- `EMAHelper` → `ExponentialMovingAverage`
- `DenoisingDiffusion` → `DenoisingDiffusionPipeline`
- `DiffusiveRestoration` → `DiffusionRestorationPipeline`

## 4) How to run (copy/paste)

### Option 1: GPU inference (Windows, recommended)

This repo already uses a CUDA venv named `.venv-gpu`.

1. Verify CUDA is visible:

```powershell
cd "c:/Users/gella/OneDrive/Desktop/LLIV"
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

2. Single-image inference:

```powershell
cd "c:/Users/gella/OneDrive/Desktop/LLIV"
.\.venv-gpu\Scripts\python.exe cli_infer.py --config configs/custom_eval.yml --resume ckpt/stage2/stage2_weight.pth.tar --input demo_data/low/IMG_5371.JPG --output results_gpu/cli/IMG_5371.png

# Higher quality (slower): more diffusion sampling steps
.\.venv-gpu\Scripts\python.exe cli_infer.py --config configs/custom_eval_quality.yml --resume ckpt/stage2/stage2_weight.pth.tar --input demo_data/low/IMG_5371.JPG --output results_gpu/cli/IMG_5371_quality.png
```

Notes:

- On 4GB GPUs, large images will OOM on full-res; the script will automatically switch to **tiled inference**.
- Tiled inference now tries larger tiles first for speed (falls back automatically if it OOMs).
- EXIF rotation is applied automatically.

3. Upload UI:

```powershell
cd "c:/Users/gella/OneDrive/Desktop/LLIV"
.\.venv-gpu\Scripts\python.exe app.py --config configs/custom_eval.yml --resume ckpt/stage2/stage2_weight.pth.tar --host 127.0.0.1 --port 7860
```

Then open: `http://127.0.0.1:7860`

Notes:

- The UI auto-saves each output under `results_ui/`.
- For large images that require tiling, use **Blend → High blend (seamless)** to reduce seams (at the cost of runtime).

### Option 2: CPU inference (works, slower)

If you’re using the CPU venv `.venv`, run the same commands but with `.\.venv\Scripts\python.exe`.

## 5) Troubleshooting

- **“Input/output alignment mismatch” on iPhone photos**
  - Cause: EXIF orientation metadata.
  - Fix: already handled in `datasets/dataset.py`, `cli_infer.py`, and `app.py` via `ImageOps.exif_transpose`.

- **CUDA out of memory (RTX 3050 4GB)**
  - Cause: full-res diffusion on 6000×4000 is too large.
  - Fix: code falls back to overlapping tiled inference in:
    - `models/restoration.py`
    - `cli_infer.py`
    - `app.py`

- **No GPU detected**
  - Make sure you are running with the CUDA-enabled environment (`.venv-gpu`) and that your installed PyTorch build is `+cu121`.
