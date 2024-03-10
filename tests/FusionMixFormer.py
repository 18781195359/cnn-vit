import torch
import torch.nn as nn
from segformer.model import segformer_b5_city
import math
import torch.nn.functional as F
import json
from timm.models.helpers import load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from vit import VisionTransformer
from decoder import pixel_decoder
from resnet import Backbone_ResNet152_in3

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        return self.relu(self.bn(self.conv(x)))


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, channels=None, image_size=None, patch_size=None):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels

        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0]//patch_size, image_size[1]//patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=16, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        head_dim = d_model // heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_rgb, x_tir):
        B, N, C = x_rgb.shape
        q = self.q(x_tir).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = self.k(x_rgb).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = self.v(x_rgb).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Fusion_Module(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fusion_layer = nn.Linear(2 * d_model, d_model)
        self.layer1 = nn.LayerNorm(d_model)
        self.sig = nn.GELU()
        self.fusin_attn = MultiHeadAttention(d_model)
        self.fusin_attn1 = MultiHeadAttention(d_model)
        self.fusin_attn2 = MultiHeadAttention(d_model)
        self.fusin_attn3 = MultiHeadAttention(d_model)
        self.drop_path = nn.Dropout(0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.mlp = FeedForward(d_model, d_model * 4, 0.1)
        self.mlp1 = FeedForward(d_model, d_model * 4, 0.1)

    def forward(self, x_rgb, x_tir, x_sum):
        # rgb_cross =  self.drop_path(self.cross(self.norm1(x_rgb), self.norm2(x_tir)))
        x_sum_att = self.drop_path(self.fusin_attn3(self.norm1(x_sum), self.norm1(x_sum)))
        x_sum = x_sum + x_sum_att
        x_sum = x_sum + self.drop_path(self.mlp1(self.norm3(x_sum)))

        rgb_fusion = self.drop_path(self.fusin_attn(self.norm4(x_rgb), self.norm5(x_tir))) + self.drop_path(
            self.fusin_attn(self.norm4(x_rgb), x_sum))
        tir_fusion = self.drop_path(self.fusin_attn2(self.norm5(x_tir), self.norm4(x_rgb))) + self.drop_path(
            self.fusin_attn(self.norm5(x_tir), x_sum))
        res = rgb_fusion + tir_fusion + x_sum
        res = res + self.drop_path(self.mlp(self.norm2(res)))

        # res = self.drop_path(self.fusin_attn3(res, res)) + res
        # res = res + self.drop_path(self.mlp1(self.norm5(res)))
        return res, rgb_fusion, tir_fusion


class Block_fusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fusion = Fusion_Module(d_model)

    def forward(self, x_rgb, x_tir):
        x_sum = x_rgb + x_tir
        for i in range(1):
            x_sum, x_rgb_one, x_tir_one = self.fusion(x_rgb, x_tir, x_sum)

        return x_sum

class FusionMixFormer(nn.Module):
    def __init__(self, d_model, image_size, patch_size, n_cls, decoder, vit, vit_tir, patch_embed_all):
        super().__init__()
        self.image_size = image_size
        self.d_model = d_model
        self.patch_size = patch_size

        # self.conv_rgb1 = ConvBnRelu(3, 64)
        # self.conv_rgb2 = ConvBnRelu(64, 128)
        # self.conv_rgb3 = ConvBnRelu(128, 256)
        # self.conv_rgb4 = ConvBnRelu(256, 512)
        # self.conv_rgb5 = ConvBnRelu(512, 1024)
        #
        # self.conv_tir1 = ConvBnRelu(3, 64)
        # self.conv_tir2 = ConvBnRelu(64, 128)
        # self.conv_tir3 = ConvBnRelu(128, 256)
        # self.conv_tir4 = ConvBnRelu(256, 512)
        # self.conv_tir5 = ConvBnRelu(512, 1024)

        (self.conv_rgb1,
         self.conv_rgb2,
         self.conv_rgb3,
         self.conv_rgb4,
         self.conv_rgb5) = Backbone_ResNet152_in3(True)

        (self.conv_tir1,
         self.conv_tir2,
         self.conv_tir3,
         self.conv_tir4,
         self.conv_tir5) = Backbone_ResNet152_in3(True)

        self.layer_config = [4, 2, 1, 1]
        self.d_model = d_model
        self.fusion = Block_fusion(d_model)
        self.patch_embed = PatchEmbedding(
            d_model, 3, image_size, patch_size
        )

        self.vit = vit
        self.vit_tir = vit_tir
        self.n_cls = n_cls
        self.patch_embed_all = patch_embed_all
        self.decoder = decoder
        self.change_dim = nn.Linear(300, 1200)

    def resize_feature(self, feature):
        feature_dim1 = feature.permute(0, 2, 1)
        return self.change_dim(feature_dim1).permute(0, 2, 1)

    def forward(self, x_rgb, x_tir):
        H, W = x_rgb.shape[2],x_rgb.shape[3]
        feature_rgb = []
        feature_tir = []
        x1 = self.conv_rgb2(self.conv_rgb1(x_rgb))
        x2 = self.conv_rgb3(x1)
        x3 = self.conv_rgb4(x2)
        x4 = self.conv_rgb5(x3)

        feature_rgb.append(x1)
        feature_rgb.append(x2)
        feature_rgb.append(x3)
        feature_rgb.append(x4)

        x1 = self.conv_tir2(self.conv_tir1(x_tir))
        x2 = self.conv_tir3(x1)
        x3 = self.conv_tir4(x2)
        x4 = self.conv_rgb5(x3)

        feature_tir.append(x1)
        feature_tir.append(x2)
        feature_tir.append(x3)
        feature_tir.append(x4)

        feature_fusion = []
        for i in range(len(feature_rgb)):
            patch_embed = self.patch_embed_all[i]
            patch_embed1 = self.patch_embed_all[i]

            feature_rgb[i] = patch_embed(feature_rgb[i])
            if feature_rgb[i].shape[1] != self.patch_embed.num_patches:
                feature_rgb[i] = self.resize_feature(feature_rgb[i])
            feature_rgb[i] = self.vit(x_rgb, feature_rgb[i], return_features=True)

            feature_tir[i] = patch_embed1(feature_tir[i])
            if feature_tir[i].shape[1] != self.patch_embed.num_patches:
                feature_tir[i] = self.resize_feature(feature_tir[i])
            feature_tir[i] = self.vit(x_tir, feature_tir[i], return_features=True)
            feature_fusion.append(self.fusion(feature_rgb[i], feature_tir[i]))

        res = 0
        for i in range(len(feature_fusion)):
            feature_fusion[i] = feature_fusion[i][:, 1:]
            class_for_decoder = feature_fusion[i][:, -self.n_cls:]
            feature_fusion[i] = feature_fusion[i][:, :- self.n_cls]
            masks = self.decoder[i](feature_fusion[i], class_for_decoder)
            masks = F.interpolate(masks, size=(H, W), mode="bilinear")
            res += masks

        return res

