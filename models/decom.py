import torch
import torch.nn as nn
import warnings
import os
import math
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DepthConv(nn.Module):
    """
    Depthwise separable convolution: spatial conv per channel, then 1x1 conv to mix channels.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        output = self.depth_conv(input)
        output = self.point_conv(output)
        return output


class ResidualBlock(nn.Module):
    """
    Simple residual block: two 3x3 convs with LeakyReLU, plus 1x1 shortcut.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        layers = []

        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*layers)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        output = self.model(x) + self.conv(x)

        return output


class UpsamplingBlock(nn.Module):
    """
    Upsampling block: ConvTranspose2d + LeakyReLU.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        output = self.relu(self.conv(x))
        return output


class FeaturesToRGBHead(nn.Module):
    """
    Reduces feature channels to 3 (RGB) via conv stack and sigmoid.
    """
    def __init__(self, channels):
        super().__init__()

        self.conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        output = torch.sigmoid(self.conv2(self.relu(self.conv1(self.relu(self.conv0(x))))))

        return output


class RGBToFeaturesStem(nn.Module):
    """
    Expands 3-channel input to high-dimensional feature tensor via conv stack.
    """
    def __init__(self, channels):
        super().__init__()

        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        output = self.conv2(self.relu(self.conv1(self.relu(self.conv0(x)))))

        return output


class FeaturePyramid(nn.Module):
    """
    Builds multi-scale feature pyramid from input image.
    Returns three levels of features.
    """
    def __init__(self, channels):
        super().__init__()

        self.convs = nn.Sequential(nn.Conv2d(3, channels, kernel_size=(5, 5), stride=(1, 1), padding=2),
                                   nn.Conv2d(channels, channels, kernel_size=(5, 5), stride=(1, 1), padding=2))

        self.block0 = ResidualBlock(channels, channels)

        self.down0 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = ResidualBlock(channels, channels * 2)

        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block2 = ResidualBlock(channels * 2, channels * 4)

        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        level0 = self.down0(self.block0(self.convs(x)))
        level1 = self.down1(self.block1(level0))
        level2 = self.down2(self.block2(level1))

        return level0, level1, level2


class ReconstructionNet(nn.Module):
    """
    Encoder-decoder for feature extraction and image reconstruction.
    If pred_fea is None: extracts features from low/high images.
    If pred_fea is given: decodes features to RGB image.
    """
    def __init__(self, channels):
        super().__init__()

        self.pyramid = FeaturePyramid(channels)

        self.channel_down = FeaturesToRGBHead(channels)
        self.channel_up = RGBToFeaturesStem(channels)

        self.block_up0 = ResidualBlock(channels * 4, channels * 4)
        self.block_up1 = ResidualBlock(channels * 4, channels * 4)
        self.up_sampling0 = UpsamplingBlock(channels * 4, channels * 2)
        self.block_up2 = ResidualBlock(channels * 2, channels * 2)
        self.block_up3 = ResidualBlock(channels * 2, channels * 2)
        self.up_sampling1 = UpsamplingBlock(channels * 2, channels)
        self.block_up4 = ResidualBlock(channels, channels)
        self.block_up5 = ResidualBlock(channels, channels)
        self.up_sampling2 = UpsamplingBlock(channels, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels, 3, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU()

    def forward(self, x, pred_fea=None):

        if pred_fea is None:
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x[:, :3, ...])
            low_fea_down8 = self.channel_down(low_fea_down8)

            high_fea_down2, high_fea_down4, high_fea_down8 = self.pyramid(x[:, 3:, ...])
            high_fea_down8 = self.channel_down(high_fea_down8)

            return low_fea_down8, high_fea_down8
        else:
            # =================low ori decoder=================
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x[:, :3, ...])

            pred_fea = self.channel_up(pred_fea)

            pred_fea_up2 = self.up_sampling0(
                self.block_up1(self.block_up0(pred_fea) + low_fea_down8))
            pred_fea_up4 = self.up_sampling1(
                self.block_up3(self.block_up2(pred_fea_up2) + low_fea_down4))
            pred_fea_up8 = self.up_sampling2(
                self.block_up5(self.block_up4(pred_fea_up4) + low_fea_down2))

            pred_img = self.conv3(self.relu(self.conv2(pred_fea_up8)))

            return pred_img


class SelfAttention(nn.Module):
    """
    Channel-mixing self-attention block (not standard spatial attention).
    """
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        qkv_tensor = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv_tensor.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attention_weights = (q @ k.transpose(-2, -1))
        attention_weights = attention_weights.softmax(dim=-1)

        attended = (attention_weights @ v)

        attended = rearrange(
            attended,
            'b head c (h w) -> b (head c) h w',
            head=self.num_heads,
            h=height,
            w=width,
        )

        attended = self.project_out(attended)
        return attended


class CrossAttention(nn.Module):
    """
    Channel-mixing cross-attention between hidden_states and context tensor.
    """
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = DepthConv(in_ch=dim, out_ch=dim)
        self.key = DepthConv(in_ch=dim, out_ch=dim)
        self.value = DepthConv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        query_features = self.query(hidden_states)
        key_features = self.key(ctx)
        value_features = self.value(ctx)

        query_layer = self.transpose_for_scores(query_features)
        key_layer = self.transpose_for_scores(key_features)
        value_layer = self.transpose_for_scores(value_features)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        return context_layer


class RetinexDecomposition(nn.Module):
    """
    Retinex-style decomposition: estimates reflectance and illumination from features using attention.
    """
    def __init__(self, channels):
        super().__init__()

        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(ResidualBlock(channels, channels),
                         ResidualBlock(channels, channels))

        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks1 = nn.Sequential(ResidualBlock(channels, channels),
                         ResidualBlock(channels, channels))

        self.cross_attention = CrossAttention(dim=channels, num_heads=8)
        self.self_attention = SelfAttention(dim=channels, num_heads=8, bias=True)

        self.conv0_1 = nn.Sequential(ResidualBlock(channels, channels),
                                     nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.conv1_1 = nn.Sequential(ResidualBlock(channels, channels),
                                     nn.Conv2d(channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=1))

    def forward(self, x):
        initial_illumination = torch.max(x, dim=1, keepdim=True)[0]
        initial_reflectance = x / initial_illumination

        reflectance_features, illumination_features = (
            self.blocks0(self.conv0(initial_reflectance)),
            self.blocks1(self.conv1(initial_illumination)),
        )

        reflectance_attended = self.cross_attention(illumination_features, reflectance_features)

        illumination_content = self.self_attention(illumination_features)

        reflectance_attended = self.conv0_1(reflectance_attended + illumination_content)
        illumination_final = self.conv1_1(illumination_features - illumination_content)

        reflectance = torch.sigmoid(reflectance_attended)
        illumination = torch.sigmoid(illumination_final)
        illumination = torch.cat([illumination for _ in range(3)], dim=1)

        return reflectance, illumination


class DecompositionReconstructionNet(nn.Module):
    """
    Top-level decomposition + reconstruction module.
    If pred_fea is None: decomposes low/high images to features and retinex outputs.
    If pred_fea is given: reconstructs enhanced image from low + features.
    """
    def __init__(self, channels=64):
        super().__init__()

        self.ReconNet = ReconstructionNet(channels)
        self.retinex = RetinexDecomposition(channels)

    def forward(self, images, pred_fea=None):

        output = {}
        # NOTE: `images` is expected to be 6-channel: (low RGB, high RGB) concatenated on channel dim.
        # - images[:, :3, ...] is treated as the low-light image
        # - images[:, 3:, ...] is treated as the high-light/reference image
        # During inference we often pass (low, low) since no paired high image exists.
        #
        # Two modes:
        # 1) pred_fea is None  -> "decompose" mode: returns dict containing low/high features + Retinex pieces.
        # 2) pred_fea is given -> "decode" mode: reconstruct RGB from predicted features (used by stage2).
        # =================decomposition low=================
        if pred_fea is None:
            low_fea_down8, high_fea_down8 = self.ReconNet(images, pred_fea=None)

            low_R, low_L = self.retinex(low_fea_down8)
            high_R, high_L = self.retinex(high_fea_down8)

            output["low_R"] = low_R
            output["low_L"] = low_L
            output["low_fea"] = low_fea_down8
            output["high_R"] = high_R
            output["high_L"] = high_L
            output["high_fea"] = high_fea_down8

        else:
            pred_img = self.ReconNet(images[:, :3, ...], pred_fea=pred_fea)
            output["pred_img"] = pred_img

        return output


# Backwards-compatible aliases (old names are kept so existing imports keep working).
Depth_conv = DepthConv
Res_block = ResidualBlock
upsampling = UpsamplingBlock
channel_down = FeaturesToRGBHead
channel_up = RGBToFeaturesStem
Self_Attention = SelfAttention
Cross_Attention = CrossAttention
Retinex_decom = RetinexDecomposition
feature_pyramid = FeaturePyramid
ReconNet = ReconstructionNet
CTDN = DecompositionReconstructionNet
