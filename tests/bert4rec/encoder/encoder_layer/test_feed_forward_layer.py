import pytest
import torch
import torch.nn as nn
from bert4rec.encoder.encoder_layer._feed_forward_layer import (
    ConfFeedForwardLayer, FeedForwardLayer)


@pytest.fixture(scope='session')
def conf_feed_forward_layer() -> ConfFeedForwardLayer:
    return ConfFeedForwardLayer(
        d_in=128,       # d_model
        d_hid=256,
        activation=nn.ReLU(inplace=True),
        name='test_feed_forward_layer',
    )

def test_feed_forward_layer_class(conf_feed_forward_layer: ConfFeedForwardLayer):
    c: ConfFeedForwardLayer = conf_feed_forward_layer

    ff: FeedForwardLayer = FeedForwardLayer(c)

    B = 4
    batch_x = torch.randn(B, c.d_in)
    out = ff(batch_x)

    assert out.shape == (B, c.d_in)
