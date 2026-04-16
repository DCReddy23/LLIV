# LLIV — Technical Deep Dive (Architecture + Code Walkthrough)

This document is a presentation-ready, code-grounded walkthrough of the LLIV repository.
It explains **what the model is**, **how data flows through it**, and **what each major module/function does**, so you can confidently present technical insights.

## 0) What this project does (in one slide)

**Goal:** Enhance low-light images.

**Approach:** A two-stage architecture:

1. **Decomposition / Retinex stage (CTDN)** extracts low-light features and can reconstruct an image from predicted features.
2. **Diffusion stage** predicts improved features via a diffusion sampler driven by a U-Net (`DiffusionUNet`). Those features are then decoded back to an image by CTDN.

**Practical usage:** You typically run **stage2 inference** using `ckpt/stage2/stage2_weight.pth.tar`.

## 1) Repository map (what to present)

### Main entrypoints

- `train.py` — stage2 training loop driver
- `evaluate.py` — dataset/filelist evaluation driver
- `cli_infer.py` — single-image inference (recommended for saving outputs)
- `app.py` — Gradio upload UI (preview + auto-save)

### Core modules

- `models/ddm.py` — diffusion wrapper + training + checkpoint loading
- `models/unet.py` — diffusion U-Net backbone
- `models/decom.py` — CTDN decomposition & reconstruction (Retinex-inspired)
- `models/restoration.py` — evaluation-time restoration and tiled inference fallback

### Data + utilities

- `datasets/dataset.py` — dataset + filelist loader
- `datasets/data_augment.py` — paired transforms (low/high kept aligned)
- `utils/sampling.py` — value transforms between `[0,1]` and `[-1,1]`
- `utils/logging.py` — image save + checkpoint I/O
- `utils/optimize.py` — optimizer factory

## 2) Data conventions (shapes + ranges)

### Image tensor ranges

- PIL images are converted to tensors in `[0, 1]` via `torchvision.transforms.functional.to_tensor`.
- Diffusion conditioning features are mapped to `[-1, 1]` using:

$$
\texttt{data\_transform}(X) = 2X - 1
$$

and mapped back using:

$$
\texttt{inverse\_data\_transform}(X) = \text{clip}\left(\frac{X+1}{2}, 0, 1\right)
$$

### Conditional vs non-conditional

Configs set `data.conditional: True`. In this repo’s inference paths, the network is called with **6 channels** by concatenating the low image with itself:

- `model_in = cat([x_cond, x_cond], dim=1)` → shape `(B, 6, H, W)`

This matches `DiffusionUNet`’s `in_channels` when conditional.

## 3) Architecture (what’s happening conceptually)

### 3.1 CTDN (decomposition + reconstruction)

CTDN is implemented in `models/decom.py` and has two main components:

- `Retinex_decom`: estimates reflectance $R$ and illumination $L$ from features using attention.
- `ReconNet`: builds a multi-scale feature pyramid and reconstructs the final RGB image.

CTDN is used in two modes:

1. **Decomposition mode** (`pred_fea is None`):
   - Takes both low and high images (6 channels)
   - Produces feature maps and retinex outputs: `low_fea`, `low_R`, `low_L`, `high_L`, etc.
2. **Reconstruction mode** (`pred_fea is not None`):
   - Takes the low image (first 3 channels) and a predicted feature tensor `pred_fea`
   - Produces an enhanced RGB output `pred_img`

### 3.2 Diffusion model (DDIM-like sampling)

Implemented in `models/ddm.py`.

- `Net` wraps:
  - `Unet` = `DiffusionUNet(config)`
  - `decom` = `CTDN()`

Sampling is performed by `Net.sample_training`, which:

- Uses `num_diffusion_timesteps` (e.g., 1000) and `num_sampling_timesteps` (e.g., 10/20/50) from the config
- Builds a coarse sampling schedule by skipping steps:

$$
\texttt{skip} = \frac{T}{S}
$$

where $T$ is diffusion timesteps and $S$ is sampling steps.

In evaluation mode, the forward pass does:

1. Run CTDN decomposition to get `low_fea`.
2. Normalize `low_fea` into `[-1, 1]`.
3. Run diffusion sampling to predict `pred_fea`.
4. Map `pred_fea` back to `[0, 1]`.
5. Run CTDN reconstruction to get the final enhanced image (`pred_img`).

### 3.3 Training objective

In training mode, `Net.forward` produces:

- `noise_output`: predicted noise from diffusion U-Net
- `e`: sampled Gaussian noise
- `pred_fea`: predicted feature
- `reference_fea`: feature target derived from the retinex decomposition

Loss in `DenoisingDiffusion.noise_estimation_loss`:

- Noise MSE: $\|\hat{\epsilon} - \epsilon\|_2^2$
- SCC feature term: $0.001 \cdot \|\texttt{pred\_fea} - \texttt{reference\_fea}\|_1$

During training, CTDN parameters are frozen:

- Any parameter name containing `"decom"` has `requires_grad = False`.

## 4) Inference paths (how you run it)

### 4.1 Single-image CLI inference (`cli_infer.py`)

**What it does:** loads one image, runs stage2, saves output.

Key behaviors:

- Fixes phone EXIF rotation using `ImageOps.exif_transpose`.
- Pads to multiples of 64 using reflect padding (model expects down/up-sampling).
- If full-resolution inference OOMs on CUDA (common for 6000×4000 on 4GB GPUs), it automatically falls back to **overlapping tiled inference**.

### 4.2 Gradio UI (`app.py`)

**What it does:** upload image → preview output → auto-save.

UI controls:

- **Quality:**
  - `Low (fast)` → sets `diffusion.num_sampling_timesteps = 10`
  - `Normal` → uses default from config (typically 20)
  - `Higher quality` → sets `diffusion.num_sampling_timesteps = 50`

- **Blend:**
  - `Normal` → faster tiling (less overlap, no context padding, standard window)
  - `High blend (seamless)` → reduced seams (more overlap, extra context padding, stronger center weighting)

Saving:

- Each UI run auto-saves into `results_ui/` as `enhanced_<timestamp>_<id>.png`.

### 4.3 Dataset/filelist evaluation (`evaluate.py` + `models/restoration.py`)

**What it does:** reads a text file of low/high pairs and writes restored outputs.

- Filelist path: `demo_data/<val_dataset>_val.txt`
- Each line: `low_path high_path`
- Output folder: `<image_folder>/<val_dataset>/...`

This path also has CUDA OOM → tiled fallback.

## 5) Tiled inference (why seams happen and what we do)

### Why seams happen

If you run a neural model on independent tiles, pixels near tile edges see different context than pixels in the center. For nonlinear models, that changes predictions and creates visible boundaries.

### Current seam-reduction strategy

Used in:

- `cli_infer.py: infer_tiled`
- `models/restoration.py: DiffusiveRestoration._restore_tiled`
- `app.py: _infer_tiled`

Techniques:

1. **Overlap + blending window**: blend overlapping outputs using a Hann window (squared in CLI/eval) so tile edges contribute less.
2. **Context padding** (CLI/eval; UI only in High blend): run each tile with extra surrounding pixels, then crop back to the tile core.
3. **Adaptive tile sizes**: try large tiles first for speed; if OOM, fall back to smaller tiles.

Trade-offs:

- Larger tile size → fewer tiles → faster (if VRAM fits)
- Larger overlap/context/window power → fewer seams → slower

## 6) Configs (what knobs matter)

Configs are YAML under `configs/`.

Important fields:

- `diffusion.num_diffusion_timesteps` — base diffusion length (e.g., 1000)
- `diffusion.num_sampling_timesteps` — number of sampling steps used at inference
  - Lower = faster, potentially worse quality
  - Higher = slower, potentially better quality

This repo includes:

- `configs/custom_eval.yml` — typical local inference config (often 20 steps)
- `configs/custom_eval_quality.yml` — higher-quality inference config (50 steps)

## 7) Checkpoints (what gets loaded)

- `ckpt/stage2/stage2_weight.pth.tar` — stage2 checkpoint used for inference
  - Loaded via `DenoisingDiffusion.load_ddm_ckpt`
  - Supports `nn.DataParallel` checkpoints by stripping `module.` prefixes
  - Uses `torch.load(..., map_location=device)` via `utils.logging.load_checkpoint`

- `ckpt/stage1/stage1_weight.pth.tar` — used during training stage2
  - Loaded in `Net.__init__` when `args.mode == 'training'`

## 8) Code walkthrough (file-by-file, function-by-function)

This section is structured as “what it is / what it takes / what it returns”.

### 8.1 Entrypoints

#### `train.py`

- `parse_args_and_config()`
  - Parses CLI args, reads `configs/<args.config>`, returns `(args, config_namespace)`.
- `dict2namespace(config_dict)`
  - Recursively converts dict to `argparse.Namespace`.
- `main()`
  - Picks device, seeds RNG, creates dataset, instantiates `DenoisingDiffusion`, calls `train()`.

#### `evaluate.py`

- `parse_args_and_config()` / `dict2namespace()`
  - Same pattern as training.
- `main()`
  - Creates dataset loaders, instantiates `DenoisingDiffusion`, wraps with `DiffusiveRestoration`, calls `restore()`.

#### `cli_infer.py`

- `pad_to_64(x)`
  - Reflect-pads BCHW tensor to multiples of 64.
- `infer_tiled(diffusion, x_cond_cpu, tile_sizes=None, overlap=128)`
  - Overlapping tiled inference with context padding + Hann blending.
- `main()`
  - Loads config + checkpoint, reads one image, runs full-res or tiled fallback, saves output.

#### `app.py`

- `load_pipeline(config_path, resume_path)`
  - Loads config, builds `DenoisingDiffusion`, loads checkpoint, caches result for the UI.
- `make_infer_fn(config_path, resume_path)`
  - Creates the Gradio callable and implements Quality/Blend logic.
- `_infer_tiled(...)`
  - Shared tiled inference implementation for UI fallback.

### 8.2 Dataset

#### `datasets/dataset.py`

- `LLdataset.get_loaders()`
  - Creates `AllWeatherDataset` for train/val and wraps with PyTorch DataLoaders.
- `AllWeatherDataset.get_images(index)`
  - Loads `low` + `high` paths from filelist, applies EXIF transpose, paired transforms, returns:
    - tensor `(6, H, W)` concatenated low/high
    - image id string

#### `datasets/data_augment.py`

Paired transforms preserve alignment between low and high images.

### 8.3 Diffusion + training

This section is intentionally **very detailed** so you can explain the code line-by-line in a presentation.

#### models/**init**.py

Purpose: convenience re-exports.

- `from models.ddm import *`
  - Makes `DenoisingDiffusion`, `Net`, etc. importable as `from models import DenoisingDiffusion`.
- `from models.restoration import *`
  - Makes `DiffusiveRestoration` importable as `from models import DiffusiveRestoration`.

#### models/ddm.py — Diffusion core + training wrapper

This file defines the **stage2 diffusion pipeline**.

##### Class: `EMAHelper`

Role: keeps an **Exponential Moving Average** (EMA) of model parameters for potentially smoother evaluation.

- `__init__(mu=0.9999)`
  - `mu` is the EMA decay; higher means slower updates.
  - Internal state: `self.shadow: dict[str, Tensor]` mapping parameter name → EMA value.

- `register(module)`
  - Walks through `module.named_parameters()` and copies each trainable parameter into `shadow`.
  - If `module` is `nn.DataParallel`, it unwraps it (`module.module`).

- `update(module)`
  - For each trainable parameter: `shadow = (1-mu)*param + mu*shadow`.
  - This is called every training step in `DenoisingDiffusion.train()`.

- `ema(module)`
  - Overwrites `module` parameters with the EMA values from `shadow`.
  - Used when you want to evaluate with EMA weights.

- `ema_copy(module)`
  - Creates a new module instance of the same type and loads current weights.
  - Applies EMA weights to that copy and returns it.
  - Handles `DataParallel` by copying the inner module and then wrapping again.

- `state_dict()` / `load_state_dict(state_dict)`
  - Save/load EMA shadow weights.

##### Function: `get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)`

Role: generate the diffusion noise schedule $\beta_t$ (length `num_diffusion_timesteps`).

Supported schedules:

- `linear`: `linspace(beta_start, beta_end, T)`
- `quad`: linear in $\sqrt{\beta}$ then squared (slower ramp at start)
- `const`: constant `beta_end`
- `jsd`: `1/t` style schedule
- `sigmoid`: sigmoid-shaped schedule between `beta_start` and `beta_end`

Output:

- `betas`: numpy array shape `(T,)`.

##### Class: `Net(nn.Module)`

Role: the **actual neural network** that the diffusion wrapper trains/evaluates.
It combines:

- `self.Unet`: diffusion U-Net (`models/unet.py`)
- `self.decom`: CTDN decomposition/reconstruction (`models/decom.py`)

Key expectations:

- Inputs are typically **6-channel** tensors `(B, 6, H, W)`.
  - In many inference paths, the code duplicates the low image to form 6 channels.

###### `__init__(args, config)`

What it does:

1. Saves `args`, `config`, and `device`.
2. Builds `DiffusionUNet(config)`.
3. Builds CTDN:
   - If `args.mode == 'training'`: loads stage1 weights into CTDN via `load_stage1(...)`.
   - Else: uses an uninitialized `CTDN()` but will later be loaded from the stage2 checkpoint via `load_ddm_ckpt()`.
4. Builds diffusion `betas` with `get_beta_schedule(...)` and stores `self.num_timesteps`.

Important note:

- `load_stage1(...)` loads `ckpt/stage1/stage1_weight.pth.tar` with a hardcoded `'cuda'` map location.
  - This is fine for GPU training but would need adjustment for CPU-only training.

###### `compute_alpha(beta, t)`

Role: compute cumulative product of alphas at timestep indices `t`.

- Prepends a 0 so index math matches the original implementation.
- Uses:

$$
\alpha_t = \prod_{i=1}^{t} (1-\beta_i)
$$

Returns:

- Tensor shaped `(B, 1, 1, 1)` for broadcasting.

###### `load_stage1(model, model_dir)`

Role: load the stage1 CTDN checkpoint from `model_dir/stage1_weight.pth.tar` and return the model.

###### `sample_training(x_cond, b, eta=0.)`

Despite the name, this is used for _sampling predicted features_ both in training and evaluation.

Inputs:

- `x_cond`: conditioning tensor (here, `low_condition_norm`) shaped `(B, C, H, W)` in `[-1, 1]`.
- `b`: `betas` tensor on the current device.
- `eta`: controls stochasticity (0 makes it more deterministic DDIM-like).

Step schedule:

- Computes `skip = T // S` where:
  - `T = num_diffusion_timesteps`
  - `S = num_sampling_timesteps`
- Uses a sequence `seq = range(0, T, skip)`.

Algorithm sketch:

1. Start from noise `x ~ N(0, I)`.
2. For each pair of timesteps `(i, j)` in reverse (`i` current, `j` next):
   - Compute `a_t` and `a_{t-1}` via `compute_alpha`.
   - Predict noise with the U-Net: `et = Unet(cat([x_cond, xt]), t)`.
   - Compute predicted clean sample `x0_t`.
   - Update to `x_{t-1}` using the DDIM-like formula with `c1`, `c2`.

Output:

- The final sample `xs[-1]`, same shape as `x`.

###### `forward(inputs)`

This function contains **two different pipelines** depending on `self.training`.

Common setup:

- `inputs`: expected shape `(B, 6, H, W)`.
- `b = self.betas.to(inputs.device)`.

Training branch (`self.training == True`):

1. Run CTDN decomposition: `output = self.decom(inputs, pred_fea=None)`.
2. Extract:
   - `low_R`, `low_L`, `low_fea`, `high_L`.
3. Conditioning tensor:
   - `low_condition_norm = utils.data_transform(low_fea)` → in `[-1,1]`.
4. Sample timesteps `t` with a symmetric trick:
   - Draw random `t` for half the batch and mirror them around `T`.
   - Helps balance early/late diffusion steps.
5. Compute diffusion mixture weight `a` from `b` and `t`.
6. Sample noise `e ~ N(0, I)`.
7. Build the “high” target in normalized space:
   - `high_input_norm = utils.data_transform(low_R * high_L)`.
8. Construct noisy input:
   - `x = high_input_norm * sqrt(a) + e * sqrt(1-a)`.
9. Predict noise with U-Net:
   - `noise_output = Unet(cat([low_condition_norm, x]), t)`.
10. Predict features with sampling:

- `pred_fea = sample_training(low_condition_norm, b)`.

11. Compute a feature reference:

- `reference_fea = low_R * (low_L ** 0.2)`.

12. Return a dict with keys used by the loss:

- `noise_output`, `e`, `pred_fea`, `reference_fea`.

Evaluation branch (`self.training == False`):

1. Run CTDN decomposition to get `low_fea`.
2. Normalize: `low_condition_norm = utils.data_transform(low_fea)`.
3. Sample predicted features: `pred_fea = sample_training(low_condition_norm, b)`.
4. Map back: `pred_fea = utils.inverse_data_transform(pred_fea)`.
5. Reconstruct RGB output using CTDN:
   - `pred_x = self.decom(inputs, pred_fea=pred_fea)["pred_img"]`.
6. Return `{ "pred_x": pred_x }`.

##### Class: `DenoisingDiffusion`

Role: training harness + checkpoint loader around `Net`.

###### `__init__(args, config)`

Creates:

- `self.model = Net(args, config)` moved to `config.device`.
- `self.ema_helper = EMAHelper()` and registers the model.
- Losses: L2 (`MSELoss`) and L1 (`L1Loss`).
- Optimizer from `utils.optimize.get_optimizer(config, model.parameters())`.

Also keeps counters `start_epoch` and `step`.

###### `load_ddm_ckpt(load_path, ema=False)`

Loads the stage2 checkpoint and applies it to `self.model`.

Key details:

- Uses `utils.logging.load_checkpoint(load_path, self.device)` → calls `torch.load(..., map_location=device)`.
- Supports `nn.DataParallel` checkpoints:
  - If keys start with `module.`, it strips that prefix before `load_state_dict`.
- `strict=True` ensures all parameters must match.
- If `ema=True`, it overwrites weights with EMA shadow.

###### `train(DATASET)`

High-level flow:

1. Builds `train_loader, val_loader = DATASET.get_loaders()`.
2. If `args.resume` exists, loads it via `load_ddm_ckpt`.
3. Freezes CTDN (“decom”) parameters:
   - Names containing `"decom"` → `requires_grad=False`.
4. Main epoch/step loop:
   - `x` is moved to device.
   - `output = self.model(x)`.
   - `loss = noise_loss + scc_loss`.
   - Backprop + optimizer step.
   - Update EMA after optimizer.
5. Every `config.training.validation_freq` steps:
   - Runs `sample_validation_patches`.
   - Saves checkpoint under `config.data.ckpt_dir/model_latest.pth.tar`.

Practical note:

- The training path expects the dataset to provide paired low/high in a single tensor `(6, H, W)`.

###### `noise_estimation_loss(output)`

Inputs:

- `output` dict produced by `Net.forward` in training.

Computes:

- `noise_loss = MSE(noise_output, e)`.
- `scc_loss = 0.001 * L1(pred_fea, reference_fea)`.

Returns: `(noise_loss, scc_loss)`.

###### `sample_validation_patches(val_loader, step)`

Saves patch outputs for monitoring training.

- Pads `x` to multiples of 64 (reflect).
- Runs `pred_x = model(x)["pred_x"]`.
- Writes images under `args.image_folder/<dataset+patch_size>/<step>/...`.

#### models/unet.py — Diffusion U-Net backbone

This is the U-Net used by diffusion to predict noise / residuals.

##### Function: `get_timestep_embedding(timesteps, embedding_dim)`

Role: create sinusoidal positional embeddings for scalar diffusion timesteps.

Inputs:

- `timesteps`: tensor shape `(B,)`.
- `embedding_dim`: integer.

Output:

- embedding tensor shape `(B, embedding_dim)`.

Mechanics:

- Uses sine/cosine pairs with exponentially-spaced frequencies (standard DDPM style).

##### Function: `nonlinearity(x)`

- Implements Swish: `x * sigmoid(x)`.

##### Function: `Normalize(in_channels)`

- Returns `GroupNorm(num_groups=32, num_channels=in_channels)`.

##### Class: `Upsample`

Role: upsample by 2×.

- `__init__(in_channels, with_conv)`
  - If `with_conv=True`, adds a `3×3` conv after nearest-neighbor upsample.
- `forward(x)`
  - `interpolate(scale_factor=2, mode='nearest')` then optional conv.

##### Class: `Downsample`

Role: downsample by 2×.

- `__init__(in_channels, with_conv)`
  - If `with_conv=True`, uses a stride-2 conv. Because PyTorch conv doesn’t support asymmetric padding directly, it manually pads (0,1,0,1).
  - If `with_conv=False`, uses `avg_pool2d`.
- `forward(x)`
  - Applies the chosen downsampling.

##### Class: `ResnetBlock`

Role: a residual block conditioned on timestep embedding (`temb`).

Key components:

- GroupNorm + Swish + conv
- Add timestep projection into the feature map
- Dropout + conv
- Optional shortcut (`conv_shortcut` or `nin_shortcut`) if channel dims change

`forward(x, temb)`:

1. Normalizes and convs `x` → `h`.
2. Projects `temb` and adds to `h` (broadcast over spatial).
3. Norm + Swish + dropout + conv.
4. Adjusts `x` if needed for channel alignment.
5. Returns `x + h`.

##### Class: `AttnBlock`

Role: spatial self-attention at a given resolution.

`forward(x)`:

1. Normalizes.
2. Computes `q,k,v` via `1×1` convs.
3. Reshapes to `(B, HW, C)` and computes attention weights with softmax.
4. Applies attention to `v` and projects back.
5. Residual add.

##### Class: `DiffusionUNet`

Role: multi-resolution U-Net with attention at one resolution level.

Configuration-driven:

- Base channels: `config.model.ch`
- Output channels: `config.model.out_ch` (3)
- Multipliers: `config.model.ch_mult` (e.g., `[1,2,3,4]`)
- Residual blocks per level: `config.model.num_res_blocks`
- Conditional input channels:
  - If `config.data.conditional` → in_channels = `2*config.model.in_channels`

Architecture:

- Timestep embedding MLP: two linear layers.
- Down path:
  - At each level: `num_res_blocks` ResnetBlocks (+ attention at level index 2)
  - Downsample between levels
- Middle:
  - ResnetBlock → AttnBlock → ResnetBlock
- Up path:
  - Mirrors down path with skip connections by concatenating with stored activations (`hs`).
  - Upsamples between levels.

`forward(x, t)`:

1. Build `temb` from `t`.
2. Downsample path: store `hs` at each block.
3. Middle blocks.
4. Upsample path: concat current with `hs.pop()` then apply blocks.
5. Normalize + Swish + final conv → output.

#### models/decom.py — CTDN decomposition & reconstruction

This file implements the Retinex-like decomposition + reconstruction network CTDN.

##### Class: `Depth_conv`

Role: depthwise-separable convolution.

- `depth_conv`: groups = in_ch (per-channel spatial conv)
- `point_conv`: `1×1` conv to mix channels

`forward(input)`:

- Applies depthwise conv then pointwise conv.

##### Class: `Res_block`

Role: simple residual block.

Structure:

- `3×3` conv → LeakyReLU → `3×3` conv
- Plus a `1×1` conv shortcut to match `out_channels`

`forward(x)` returns `model(x) + conv(x)`.

##### Class: `upsampling`

Role: upsample by transpose convolution.

- `ConvTranspose2d(stride=2)` then LeakyReLU.

##### Class: `channel_down`

Role: compress features down to RGB.

- Uses a small conv stack:
  - `(4C → 2C → C → 3)` with LeakyReLU between
- Final sigmoid ensures output is in `[0,1]`.

##### Class: `channel_up`

Role: expand a 3-channel tensor to a high-dimensional feature tensor.

- `(3 → C → 2C → 4C)` conv stack.

##### Class: `feature_pyramid`

Role: encode an RGB image into multi-scale feature maps.

`forward(x)`:

1. Two `5×5` convs produce base features.
2. `block0` then `down0` → level0 (downsampled)
3. `block1` then `down1` → level1
4. `block2` then `down2` → level2 (deepest)

Returns `(level0, level1, level2)`.

##### Class: `ReconNet`

Role: encode low/high into deep features, and decode predicted features into an image.

Key methods:

- `forward(x, pred_fea=None)` has two modes.

Mode A: `pred_fea is None` (feature extraction)

- Splits the 6-channel input into:
  - `x[:, :3]` low image
  - `x[:, 3:]` high image
- Runs `feature_pyramid` on each, takes the deepest level.
- Uses `channel_down` to map deep features to a 3-channel representation.
- Returns `(low_fea_down8, high_fea_down8)`.

Mode B: `pred_fea provided` (reconstruction)

- Builds pyramid features for low image.
- Expands `pred_fea` using `channel_up`.
- Decodes upward while adding skip connections from the pyramid.
- Outputs `pred_img` via final conv layers.

##### Class: `Self_Attention`

Role: **channel-mixing attention** within a single feature map.

This block is not the typical “spatial attention over $H\times W$ tokens”. In this implementation:

- Q/K/V are produced by `1×1` conv then depthwise `3×3` conv.
- The tensor is rearranged to `(B, head, C_head, HW)`.
- `q` and `k` are L2-normalized over the **spatial dimension** (`HW`).
- The attention matrix is computed as `attn = softmax(q @ k^T)` which yields shape `(B, head, C_head, C_head)`.
  - That means it mixes **channels within each head**, not pixels.
- The attended output is `out = attn @ v` → `(B, head, C_head, HW)` then reshaped back to `(B, C, H, W)`.
- Final `1×1` conv projects back to `dim`.

##### Class: `Cross_Attention`

Role: cross-attention-style **channel mixing** between `hidden_states` and a context tensor `ctx`.

Important implementation detail: this is not a standard “multi-head attention over flattened tokens”.

- Query/Key/Value are produced by `Depth_conv` and remain 4D tensors `(B, C, H, W)`.
- `transpose_for_scores(x)` simply does `x.permute(0, 2, 1, 3)` → `(B, H, C, W)`.
  - There is no explicit reshape that splits channels into `num_heads`.
  - `num_heads` only influences the scaling term via `attention_head_size = dim / num_heads`.
- Attention is computed as:
  - `attention_scores = (B, H, C, W) @ (B, H, W, C)` → `(B, H, C, C)`
  - softmax over the last dim → channel-to-channel mixing per row index `H`.
- The result is applied to `value_layer` and permuted back to `(B, C, H, W)`.

##### Class: `Retinex_decom`

Role: estimate reflectance $R$ and illumination $L$ (Retinex-style).

`forward(x)`:

1. Initialize illumination as channel-wise max: `init_illumination = max(x, dim=1)`.
2. Initialize reflectance as ratio: `init_reflectance = x / init_illumination`.
3. Encode reflectance and illumination separately through conv+ResBlocks.
4. Cross-attend illumination to reflectance: `Reflectance_final = cross_attention(Illumination, Reflectance)`.
5. Self-attend illumination: `Illumination_content = self_attention(Illumination)`.
6. Final conv heads produce:
   - 3-channel reflectance logits
   - 1-channel illumination logits
7. Apply sigmoid to get `R` and `L` in `[0,1]`.
8. Repeat `L` across 3 channels so it can multiply RGB.

Returns `(R, L)`.

##### Class: `CTDN`

Role: top-level decomposition + reconstruction module.

- Contains:
  - `ReconNet`
  - `Retinex_decom`

`forward(images, pred_fea=None)`:

Mode A (decomposition, `pred_fea is None`):

1. `ReconNet(images)` returns deep low/high features.
2. Run `retinex` on both low/high features.
3. Returns a dict:
   - `low_R`, `low_L`, `low_fea`
   - `high_R`, `high_L`, `high_fea`

Mode B (reconstruction, `pred_fea provided`):

1. Reconstruct enhanced image from the low image and predicted features:
   - `pred_img = ReconNet(images[:, :3], pred_fea=pred_fea)`
2. Returns `{ "pred_img": pred_img }`.

#### models/restoration.py — Evaluation restoration + tiled fallback

This file defines `DiffusiveRestoration`, used by `evaluate.py`.

##### Class: `DiffusiveRestoration`

Role: run evaluation on a DataLoader and save images to disk.

###### `__init__(diffusion, args, config)`

- Stores references.
- If `args.resume` exists, loads checkpoint via `diffusion.load_ddm_ckpt(..., ema=False)`.
- Sets `diffusion.model.eval()`.

###### `restore(val_loader)`

For each batch `(x, y)`:

1. `x` comes from dataset as concatenated `(low, high)` → shape `(B, 6, H, W)`.
2. Uses only the low image as conditioning: `x_cond_cpu = x[:, :3]`.
3. Tries full-res inference:
   - Moves `x_cond` to GPU.
   - Pads to multiples of 64 (reflect).
   - Calls model with duplicated conditioning: `cat([x_cond, x_cond])`.
4. If CUDA OOM:
   - Clears CUDA cache.
   - Runs `_restore_tiled` (overlapping tiles).
5. Saves output using `utils.logging.save_image(...)` under:
   - `<args.image_folder>/<config.data.val_dataset>/<img_id>`

###### `_restore_tiled(x_cond_cpu, h, w)`

Purpose: run overlapping tiled inference to reduce GPU memory.

Core ideas:

- Keeps `x_cond_cpu` on CPU and only moves each tile to GPU.
- Uses overlap blending with a window so tile edges are down-weighted.
- Adds **context padding** around each tile:
  - infer a larger tile `(tile + context)`
  - crop back to the core tile

Key helpers:

- `pad_to_64(tile)` reflect-pads a tile to multiples of 64.
- `hann_window_2d(th, tw)` returns a **squared** Hann window (edges get very small weight).

Loop structure:

1. Choose `tile_size` from `[1536, 1280, 1024, 768, 512, 384, 256]` (largest first).
2. Compute `stride = tile_size - overlap`.
3. Iterate `y0, x0` grid; for each tile:
   - Compute extended bounds `y0e..y1e`, `x0e..x1e` using `context = max(64, overlap//2)`.
   - Run model on extended tile.
   - Crop to core tile region.
   - Blend into `out` using window; accumulate weights.
4. Return `out / weight` clipped to `[0,1]`.

### 8.4 Utilities (brief)

- `utils/sampling.py` — range transforms between `[0,1]` and `[-1,1]`
- `utils/logging.py` — image save + checkpoint I/O
- `utils/optimize.py` — optimizer factory
