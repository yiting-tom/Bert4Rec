import pytest
import torch
from bert4rec.encoder._encoder import ConfEncoder, Encoder


@pytest.mark.order(5)
def test_encoder_class(conf_encoder: ConfEncoder):
    
    c: ConfEncoder = conf_encoder

    encoder = Encoder(c)

    B, S = 4, 128

    batch_x: torch.tensor = torch.randn(B, S, c.conf_encoder_layer.conf_multihead_attn.d_model)

    out = encoder(batch_x=batch_x, attn_bias=None)

    assert out.shape == (B, S, c.conf_encoder_layer.conf_multihead_attn.d_model)
