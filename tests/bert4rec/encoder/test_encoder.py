import pytest
import torch
from bert4rec.encoder._encoder import ConfEncoder, Encoder
from bert4rec.encoder.encoder_layer import ConfEncoderLayer

from .encoder_layer.test_encoder_layer import *


@pytest.fixture(scope='session')
def conf_encoder(conf_encoder_layer: ConfEncoderLayer) -> ConfEncoder:
    return ConfEncoder(
        n_layer=4,
        conf_encoder_layer=conf_encoder_layer,
        name='test_encoder',
    )

def test_encoder_class(conf_encoder: ConfEncoder):
    
    c: ConfEncoder = conf_encoder

    encoder = Encoder(c)

    B, S = 4, 128

    batch_x: torch.tensor = torch.randn(B, S, c.conf_encoder_layer.conf_multihead_attn.d_model)

    out = encoder(batch_x=batch_x, attn_bias=None)
    print(out.shape)
