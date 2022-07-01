import pytest
import torch
from bert4rec.encoder.encoder_layer._encoder_layer import (ConfEncoderLayer,
                                                           EncoderLayer)

from .test_attention import conf_multi_head_attention
from .test_drop_residual_normalize_layer import \
    conf_drop_residual_normalize_layer
from .test_feed_forward_layer import conf_feed_forward_layer


@pytest.fixture(scope='session')
def conf_encoder_layer(
    conf_multi_head_attention,
    conf_drop_residual_normalize_layer,
    conf_feed_forward_layer,
    ) -> ConfEncoderLayer:

    return ConfEncoderLayer(
        conf_multihead_attn=conf_multi_head_attention,
        conf_drop_residual_norm_layer=conf_drop_residual_normalize_layer,
        conf_feed_forward_layer=conf_feed_forward_layer,
        name='test_encoder_layer',
    )
    

def test_encoder_layer_class(conf_encoder_layer: ConfEncoderLayer):
    c: ConfEncoderLayer = conf_encoder_layer

    el = EncoderLayer(c)

