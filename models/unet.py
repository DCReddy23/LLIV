import math
import torch
import torch.nn as nn
import torch.nn.functional

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Builds sinusoidal timestep embeddings for diffusion models (matches DDPM/transformer style).
    Input: timesteps (B,), embedding_dim (int).
    Output: (B, embedding_dim) tensor.
    """
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_embedding_dim = embedding_dim // 2
    frequency_log_scale = math.log(10000) / (half_embedding_dim - 1)
    frequencies = torch.exp(torch.arange(half_embedding_dim, dtype=torch.float32) * -frequency_log_scale)
    frequencies = frequencies.to(device=timesteps.device)

    angular_speeds = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angular_speeds), torch.cos(angular_speeds)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))
    return embedding


def nonlinearity(x):
    """
    Swish activation: x * sigmoid(x).
    """
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    """
    Returns GroupNorm layer with 32 groups for given channel count.
    """
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    """
    Upsamples input by 2x (nearest neighbor), with optional 3x3 conv.
    """
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsamples input by 2x (stride-2 conv or avg pool).
    """
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            padding = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, padding, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with optional shortcut and timestep embedding conditioning.
    """
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = nonlinearity(hidden)
        hidden = self.conv1(hidden)

        hidden = hidden + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        hidden = self.norm2(hidden)
        hidden = nonlinearity(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + hidden


class AttnBlock(nn.Module):
    """
    Self-attention block over spatial positions (applies at one U-Net resolution).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        normalized_features = x
        normalized_features = self.norm(normalized_features)
        q = self.q(normalized_features)
        k = self.k(normalized_features)
        v = self.v(normalized_features)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        attention_weights = torch.bmm(q, k)     # b,hw,hw
        attention_weights = attention_weights * (int(c) ** (-0.5))
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        attention_weights = attention_weights.permute(0, 2, 1)   # b,hw,hw
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        attended_features = torch.bmm(v, attention_weights)
        attended_features = attended_features.reshape(b, c, h, w)

        attended_features = self.proj_out(attended_features)

        return x + attended_features


class DiffusionUNet(nn.Module):
    """
    Multi-resolution U-Net backbone for diffusion. Supports conditional input, attention, and timestep embedding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level == 2:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level == 2:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        """
        Forward pass: processes input x with timestep embedding t through U-Net.
        Returns predicted noise/residual.
        """
        # assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        timestep_embedding = get_timestep_embedding(t, self.ch)
        timestep_embedding = self.temb.dense[0](timestep_embedding)
        timestep_embedding = nonlinearity(timestep_embedding)
        timestep_embedding = self.temb.dense[1](timestep_embedding)

        # downsampling
        skip_connections = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hidden_states = self.down[i_level].block[i_block](skip_connections[-1], timestep_embedding)
                if len(self.down[i_level].attn) > 0:
                    hidden_states = self.down[i_level].attn[i_block](hidden_states)
                skip_connections.append(hidden_states)
            if i_level != self.num_resolutions-1:
                skip_connections.append(self.down[i_level].downsample(skip_connections[-1]))

        # middle
        hidden_states = skip_connections[-1]
        hidden_states = self.mid.block_1(hidden_states, timestep_embedding)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states, timestep_embedding)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                hidden_states = self.up[i_level].block[i_block](
                    torch.cat([hidden_states, skip_connections.pop()], dim=1), timestep_embedding)
                if len(self.up[i_level].attn) > 0:
                    hidden_states = self.up[i_level].attn[i_block](hidden_states)
            if i_level != 0:
                hidden_states = self.up[i_level].upsample(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = nonlinearity(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states
