import pytest
import torch
from bert4rec.encoder.encoder_layer._encoder_layer import (ConfEncoderLayer,
                                                           EncoderLayer)


@pytest.mark.order(4)
def test_encoder_layer_class(conf_encoder_layer: ConfEncoderLayer):
    c: ConfEncoderLayer = conf_encoder_layer

    el = EncoderLayer(c)

    batch_x = torch.randn(4, 128, c.conf_multihead_attn.d_model)

    out = el(batch_x=batch_x, attn_bias=None)

    assert out.shape == (4, 128, c.conf_multihead_attn.d_model)
