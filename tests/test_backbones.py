import torch

from segformer.backbones import mit_b1
from segformer.model import segformer_b5_city
def test_mit_b1():
    model = segformer_b5_city(True, True)
    backbone = model.backbone

    # backbone = mit_b1()
    x = torch.randn(2, 3, 480, 640)
    y = backbone(x)
    assert len(y) == 4
    assert y[0].shape == (2, 64, 120, 160)
    assert y[1].shape == (2, 128, 60, 80)
    assert y[2].shape == (2, 320, 30, 40)
    assert y[3].shape == (2, 512, 15, 20)

