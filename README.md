# LLIV

Low-light image enhancement (stage2 diffusion restoration) with optional upload UI.

Full code map + deeper explanations: [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Quickstart (Windows)

### 1) GPU single-image inference (recommended)

This repo uses a CUDA virtualenv at `.venv-gpu`.

Run on the included demo image:

```powershell
cd "c:/Users/gella/OneDrive/Desktop/LLIV"

.\.venv-gpu\Scripts\python.exe cli_infer.py --config configs/custom_eval.yml --resume ckpt/stage2/stage2_weight.pth.tar --input demo_data/low/IMG_5371.JPG --output results_gpu/cli/IMG_5371.png

# Higher quality (slower): more diffusion sampling steps
.\.venv-gpu\Scripts\python.exe cli_infer.py --config configs/custom_eval_quality.yml --resume ckpt/stage2/stage2_weight.pth.tar --input demo_data/low/IMG_5371.JPG --output results_gpu/cli/IMG_5371_quality.png
```

Output:

- `results_gpu/cli/IMG_5371.png`

Notes:

- Large phone photos can exceed 4GB VRAM; the code automatically falls back to tiled inference on CUDA OOM.
- EXIF orientation is handled (important for iPhone images).

### 2) Upload UI (Gradio)

One-command launcher:

```powershell
cd "c:/Users/gella/OneDrive/Desktop/LLIV"
.\run_ui.bat
```

Custom port (optional):

```powershell
.\run_ui.bat 7861
```

Open:

- http://127.0.0.1:7860

UI options:

- **Quality:** `Low (fast)` / `Normal` / `Higher quality`
- **Blend:** `Normal` (faster, may show seams) / `High blend (seamless)` (slower, fewer seams)

Saved outputs:

- Each run is auto-saved under `results_ui/` as `enhanced_<timestamp>_<id>.png`.

### 3) Dataset/filelist evaluation (optional)

If you want the original dataset-style evaluation path:

```powershell
cd "c:/Users/gella/OneDrive/Desktop/LLIV"
.\.venv-gpu\Scripts\python.exe evaluate.py --config configs/custom_eval.yml --resume ckpt/stage2/stage2_weight.pth.tar --image_folder results_gpu
```

This reads `demo_data/custom_val.txt` and writes under `results_gpu/custom/`.
