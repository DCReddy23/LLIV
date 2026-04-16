import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models')
    # Default to the local demo/eval config (uses demo_data/*_train.txt and demo_data/*_val.txt).
    # Use --config unsupervised.yml if you have the full external dataset referenced there.
    parser.add_argument("--config", default='custom_eval.yml', type=str,
                        help="Config filename under configs/ (e.g., custom_eval.yml)")
    parser.add_argument('--mode', type=str, default='evaluation', help='training or evaluation')
    parser.add_argument('--resume', default='ckpt/stage2/stage2_weight.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (auto selects CUDA if available)",
    )
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (--device cuda) but is not available. "
                "Verify NVIDIA drivers and that you installed a CUDA-enabled PyTorch build."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))
    config.device = device

    if device.type == "cuda":
        cudnn.benchmark = True
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    print("=> using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(val_loader)


if __name__ == '__main__':
    main()
