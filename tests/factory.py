from tests.FusionMixFormer import FusionMixFormer
import json
from timm.models.helpers import load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from tests.vit import VisionTransformer
from tests.decoder import pixel_decoder
from tests.FusionMixFormer import PatchEmbedding

def get_vit(cfg_model):
    backbone = cfg_model.pop("backbone")
    default_cfg = default_cfgs[backbone]
    default_cfg["input_size"] = (
        3, 480, 640
    )
    model = VisionTransformer(**cfg_model)
    load_custom_pretrained(model, default_cfg)
    cfg_model["backbone"] = backbone
    return model.to('cuda:0')

def get_decoder(cfg_model):
    decoder_cfg = dict(cfg_model)
    decoder_cfg['high'] = 120
    decoder_cfg['width'] = 160
    decoder1 = pixel_decoder(**decoder_cfg)

    decoder_cfg['high'] = 60
    decoder_cfg['width'] = 80
    decoder2 = pixel_decoder(**decoder_cfg)

    decoder_cfg['high'] = 30
    decoder_cfg['width'] = 40
    decoder3 = pixel_decoder(**decoder_cfg)

    decoder_cfg['high'] = 15
    decoder_cfg['width'] = 20
    decoder4 = pixel_decoder(**decoder_cfg)

    return decoder1.to('cuda:0'), decoder2.to('cuda:0'), decoder3.to('cuda:0'), decoder4.to('cuda:0')

def get_change_dim(d_model, image_size):
    patch1 = PatchEmbedding(d_model, 256, image_size, 4)
    patch2 = PatchEmbedding(d_model, 512, image_size, 2)
    patch3 = PatchEmbedding(d_model, 1024, image_size, 1)
    patch4 = PatchEmbedding(d_model, 2048, image_size, 1)

    return patch1.to('cuda:0'), patch2.to('cuda:0'), patch3.to('cuda:0'), patch4.to('cuda:0')

def get_model():
    with open("configs/TwinViTSeg.json", 'r') as fp:
        cfg_model = json.load(fp)
    cfg_model["image_size"] = list(cfg_model["image_size"].split(" "))
    cfg_model["image_size"][0] = int(cfg_model["image_size"][0])
    cfg_model["image_size"][1] = int(cfg_model["image_size"][1])

    d_model = cfg_model["d_model"]
    image_size = cfg_model["image_size"]
    patch_size = cfg_model["patch_size"]
    n_cls = cfg_model["n_cls"]
    vit_rgb = get_vit(cfg_model)
    vit_tir = get_vit(cfg_model)
    cfg_model.pop('backbone')
    decoder = get_decoder(cfg_model)
    patch_embed_all = get_change_dim(d_model, image_size)
    model = FusionMixFormer(d_model, image_size, patch_size, n_cls,decoder, vit_rgb, vit_tir, patch_embed_all)

    return model