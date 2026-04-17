import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import OrderedDict
import utils
from models.unet import DiffusionUNet
from models.decom import DecompositionReconstructionNet


class ExponentialMovingAverage(object):
    """
    Maintains an exponential moving average (EMA) of model parameters for evaluation stability.
    """
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    Returns a beta schedule (noise schedule) for diffusion based on the chosen strategy.
    Supported: 'linear', 'quad', 'const', 'jsd', 'sigmoid'.
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class LatentRetinexDiffusionModel(nn.Module):
    """
    Main model: wraps DiffusionUNet and DecompositionReconstructionNet.
    Handles both training and inference pipelines.
    """
    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)
        if self.args.mode == 'training':
            self.decom = self.load_stage1(DecompositionReconstructionNet(), 'ckpt/stage1')
        else:
            self.decom = DecompositionReconstructionNet()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        """
        Compute cumulative product of (1-beta) up to timestep t for diffusion process.
        """
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        alpha_cumprod = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return alpha_cumprod

    @staticmethod
    def load_stage1(model, model_dir):
        """
        Loads stage1 CTDN weights from checkpoint for use in training.
        """
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'stage1_weight.pth.tar'), 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    def sample_training(self, x_cond, b, eta=0.):
        """
        DDIM-like sampling: generates predicted features from conditioning tensor.
        Used in both training and inference.
        """
        stride = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        sampling_timesteps = range(0, self.config.diffusion.num_diffusion_timesteps, stride)
        batch_size, channels, height, width = x_cond.shape
        next_timesteps = [-1] + list(sampling_timesteps[:-1])

        sample = torch.randn(batch_size, channels, height, width, device=self.device)
        samples = [sample]
        for i, j in zip(reversed(sampling_timesteps), reversed(next_timesteps)):
            timestep = (torch.ones(batch_size) * i).to(sample.device)
            next_timestep = (torch.ones(batch_size) * j).to(sample.device)
            alpha_t = self.compute_alpha(b, timestep.long())
            alpha_next_t = self.compute_alpha(b, next_timestep.long())
            current_sample = samples[-1].to(sample.device)

            predicted_noise = self.Unet(torch.cat([x_cond, current_sample], dim=1), timestep)
            predicted_clean = (current_sample - predicted_noise * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            ddim_sigma = eta * ((1 - alpha_t / alpha_next_t) * (1 - alpha_next_t) / (1 - alpha_t)).sqrt()
            ddim_coeff = ((1 - alpha_next_t) - ddim_sigma ** 2).sqrt()
            next_sample = (
                alpha_next_t.sqrt() * predicted_clean
                + ddim_sigma * torch.randn_like(sample)
                + ddim_coeff * predicted_noise
            )
            samples.append(next_sample.to(sample.device))

        return samples[-1]

    def forward(self, inputs):
        """
        Forward pass for both training and evaluation.
        Training: returns noise prediction, sampled noise, predicted features, and reference features.
        Eval: returns predicted enhanced image.
        """
        data_dict = {}

        betas = self.betas.to(inputs.device)

        if self.training:
            decom_output = self.decom(inputs, pred_fea=None)
            low_reflectance = decom_output["low_R"]
            low_illumination = decom_output["low_L"]
            low_features = decom_output["low_fea"]
            high_illumination = decom_output["high_L"]

            low_condition_normalized = utils.data_transform(low_features)

            timestep_indices = torch.randint(
                low=0,
                high=self.num_timesteps,
                size=(low_condition_normalized.shape[0] // 2 + 1,),
            ).to(self.device)
            timestep_indices = torch.cat(
                [timestep_indices, self.num_timesteps - timestep_indices - 1],
                dim=0,
            )[:low_condition_normalized.shape[0]].to(inputs.device)

            alpha_cumprod = (1 - betas).cumprod(dim=0).index_select(0, timestep_indices).view(-1, 1, 1, 1)
            noise = torch.randn_like(low_condition_normalized)

            high_input_normalized = utils.data_transform(low_reflectance * high_illumination)
            noised_high_input = high_input_normalized * alpha_cumprod.sqrt() + noise * (1.0 - alpha_cumprod).sqrt()

            predicted_noise = self.Unet(
                torch.cat([low_condition_normalized, noised_high_input], dim=1),
                timestep_indices.float(),
            )

            predicted_features = self.sample_training(low_condition_normalized, betas)
            predicted_features = utils.inverse_data_transform(predicted_features)
            reference_features = low_reflectance * torch.pow(low_illumination, 0.2)

            data_dict["noise_output"] = predicted_noise
            data_dict["e"] = noise

            data_dict["pred_fea"] = predicted_features
            data_dict["reference_fea"] = reference_features

        else:
            decom_output = self.decom(inputs, pred_fea=None)
            low_features = decom_output["low_fea"]
            low_condition_normalized = utils.data_transform(low_features)

            predicted_features = self.sample_training(low_condition_normalized, betas)
            predicted_features = utils.inverse_data_transform(predicted_features)
            predicted_image = self.decom(inputs, pred_fea=predicted_features)["pred_img"]
            data_dict["pred_x"] = predicted_image

        return data_dict


class DenoisingDiffusionPipeline(object):
    """
    Training and inference harness for the diffusion model. Handles checkpointing, optimizer, and validation.
    """
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = LatentRetinexDiffusionModel(args, config)
        self.model.to(self.device)

        self.ema_helper = ExponentialMovingAverage()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        """
        Loads a stage2 checkpoint (optionally EMA weights) into the model. Handles DataParallel checkpoints.
        """
        checkpoint = utils.logging.load_checkpoint(load_path, self.device)

        state_dict = checkpoint['state_dict']
        # Handle checkpoints saved from nn.DataParallel (keys prefixed with 'module.').
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[len('module.'):]] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict, strict=True)
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        """
        Main training loop for diffusion model. Handles freezing decomposition module, optimizer, validation, and checkpointing.
        """
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for name, param in self.model.named_parameters():
            if "decom" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, scc_loss = self.noise_estimation_loss(output)
                loss = noise_loss + scc_loss

                data_time += time.time() - data_start

                if self.step % 10 == 0:
                    print("step:{}, noise_loss:{:.5f} scc_loss:{:.5f} time:{:.5f}".
                          format(self.step, noise_loss.item(),
                                 scc_loss.item(), data_time / (i + 1)))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({'step': self.step,
                                                   'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))

    def noise_estimation_loss(self, output):
        """
        Computes noise prediction loss (MSE) and SCC feature loss (L1) for training.
        """
        pred_fea, reference_fea = output["pred_fea"], output["reference_fea"]
        noise_output, e = output["noise_output"], output["e"]
        # ==================noise loss==================
        noise_loss = self.l2_loss(noise_output, e)
        # ==================scc loss==================
        scc_loss = 0.001 * self.l1_loss(pred_fea, reference_fea)

        return noise_loss, scc_loss

    def sample_validation_patches(self, val_loader, step):
        """
        Runs validation: saves predicted patches for visual monitoring during training.
        """
        image_folder = os.path.join(self.args.image_folder,
                                    self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()

        with torch.no_grad():
            print('Performing validation at step: {}'.format(step))
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')
                pred_x = self.model(x.to(self.device))["pred_x"][:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), '{}'.format(y[0])))


# Backwards-compatible aliases (old names are kept so existing imports keep working).
EMAHelper = ExponentialMovingAverage
Net = LatentRetinexDiffusionModel
DenoisingDiffusion = DenoisingDiffusionPipeline
