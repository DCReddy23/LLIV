import torch
import numpy as np
import utils
import os
import time
import torch.nn.functional as F


class DiffusionRestorationPipeline:
    """
    Evaluation-time restoration harness. Handles checkpoint loading, full-res and tiled inference, and saving outputs.
    """
    def __init__(self, diffusion, args, config):
        """
        Initializes restoration wrapper, loads checkpoint if present, sets model to eval mode.
        """
        super().__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained model path is missing!')

    def restore(self, val_loader):
        """
        Runs restoration on a validation DataLoader. Handles OOM fallback to tiled inference and saves results.
        """
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        os.makedirs(image_folder, exist_ok=True)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):

                # Keep the full-resolution input on CPU and only move
                # tiles to GPU if we need a low-memory fallback.
                conditioning_image_cpu = x[:, :3, :, :].contiguous()
                _, _, height, width = conditioning_image_cpu.shape

                try:
                    # Fast path: run full image on the configured device.
                    x_cond = conditioning_image_cpu.to(self.diffusion.device)
                    img_h_64 = int(64 * np.ceil(height / 64.0))
                    img_w_64 = int(64 * np.ceil(width / 64.0))
                    x_cond = F.pad(x_cond, (0, img_w_64 - width, 0, img_h_64 - height), 'reflect')

                    start_time = time.time()
                    pred_x = self.diffusion.model(torch.cat((x_cond, x_cond), dim=1))["pred_x"][:, :, :height, :width]
                    end_time = time.time()
                except torch.OutOfMemoryError:
                    if self.diffusion.device.type != "cuda":
                        raise

                    torch.cuda.empty_cache()
                    print(
                        f"CUDA OOM on full-res {height}x{width}; falling back to tiled inference...",
                        flush=True,
                    )
                    start_time = time.time()
                    pred_x = self._restore_tiled(conditioning_image_cpu, h=height, w=width)
                    end_time = time.time()

                utils.logging.save_image(pred_x, os.path.join(image_folder, f"{y[0]}"))
                print(f"processing image {y[0]}, time={end_time - start_time}")

    def _restore_tiled(self, x_cond_cpu: torch.Tensor, *, h: int, w: int) -> torch.Tensor:
        """
        Runs restoration in overlapping tiles to reduce GPU memory usage. Blends tiles to reduce seams.
        Returns a CPU tensor shaped (B, 3, H, W) in [0, 1].
        """
        """Run restoration in overlapping tiles to reduce GPU memory usage.

        Returns a CPU tensor shaped (B, 3, H, W) in [0, 1].
        """

        def pad_to_64(tile: torch.Tensor):
            _, _, th, tw = tile.shape
            th_64 = int(64 * np.ceil(th / 64.0))
            tw_64 = int(64 * np.ceil(tw / 64.0))
            tile = F.pad(tile, (0, tw_64 - tw, 0, th_64 - th), 'reflect')
            return tile, th, tw

        def hann_window_2d(th: int, tw: int) -> torch.Tensor:
            wy = torch.hann_window(th, periodic=False)
            wx = torch.hann_window(tw, periodic=False)
            return ((wy[:, None] * wx[None, :]).clamp(min=1e-6)) ** 2

        # Start reasonably large and shrink if needed.
        # Try larger tiles first for speed; fall back on OOM.
        tile_sizes = [1536, 1280, 1024, 768, 512, 384, 256]
        overlap = 128
        window_cache: dict[tuple[int, int], torch.Tensor] = {}

        context = max(64, overlap // 2)

        batch_size = x_cond_cpu.shape[0]

        for tile_size in tile_sizes:
            stride = max(64, tile_size - overlap)
            try:
                out = torch.zeros((batch_size, 3, h, w), dtype=torch.float32)
                weight = torch.zeros((1, 1, h, w), dtype=torch.float32)

                ys = list(range(0, h, stride))
                xs = list(range(0, w, stride))
                if ys[-1] + tile_size < h:
                    ys.append(h - tile_size)
                if xs[-1] + tile_size < w:
                    xs.append(w - tile_size)
                ys = sorted(set(max(0, y0) for y0 in ys))
                xs = sorted(set(max(0, x0) for x0 in xs))

                for y0 in ys:
                    print(f"tiled inference: y={y0}/{h}", flush=True)
                    for x0 in xs:
                        y1 = min(h, y0 + tile_size)
                        x1 = min(w, x0 + tile_size)

                        y0e = max(0, y0 - context)
                        x0e = max(0, x0 - context)
                        y1e = min(h, y1 + context)
                        x1e = min(w, x1 + context)

                        tile_cpu = x_cond_cpu[:, :, y0e:y1e, x0e:x1e]
                        tile = tile_cpu.to(self.diffusion.device)
                        tile, th, tw = pad_to_64(tile)

                        pred = self.diffusion.model(torch.cat((tile, tile), dim=1))["pred_x"][:, :, :th, :tw]
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
                            win = hann_window_2d(out_h, out_w).unsqueeze(0).unsqueeze(0)
                            window_cache[key] = win

                        out[:, :, y0:y1, x0:x1] += pred_crop * win
                        weight[:, :, y0:y1, x0:x1] += win

                out = out / weight
                out = out.clamp(0, 1)
                return out
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"Still OOM with tile_size={tile_size}; trying smaller tiles...", flush=True)

        raise torch.OutOfMemoryError("Unable to run tiled inference within GPU memory limits.")


# Backwards-compatible alias
DiffusiveRestoration = DiffusionRestorationPipeline



