import torch
import torch.nn as nn
from tests.vit import init_weights
from tests.vit import Block
from timm.models.layers import trunc_normal_
from einops import rearrange

class pixel_decoder(nn.Module):
    def __init__(self, image_size, patch_size, n_layers, d_model, d_ff, n_heads, n_cls, dropout=0.1, drop_path_rate=0.0, distilled=False, channels=3, high=None, width=None):
        super(pixel_decoder, self).__init__()
        self.image_size = image_size
        self.d_encoder = d_model
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.high = high

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(self.d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)
        self.recover = nn.Parameter(torch.randn(1200, high * width))

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)
        trunc_normal_(self.recover, std=0.02)


    def forward(self, x, cls_emb_fir):
        GS = self.high

        x = self.proj_dec(x)
        cls_emb = cls_emb_fir
        x = torch.cat((x, cls_emb), 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks).permute(0, 2, 1) @ self.recover
        masks = masks.permute(0, 2, 1)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks